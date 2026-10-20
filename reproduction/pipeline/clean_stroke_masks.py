"""Conservatively clean predicted stroke masks before manual review and VS."""
import argparse
import colorsys
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from common import arrays, digest, gray, iou, output_dir, preview, records, require, write_json, write_records


def component_count(mask, minimum=1):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return sum(int(value) >= minimum for value in stats[1:count, cv2.CC_STAT_AREA])


def overlay(image, masks, removed=None, added=None):
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.float32)
    colors = [np.asarray(colorsys.hsv_to_rgb(i / max(1, len(masks)), .85, 1))[::-1] * 255
              for i in range(len(masks))]
    for index, (mask, color) in enumerate(zip(masks, colors)):
        mask = mask.astype(bool)
        canvas[mask] = .35 * canvas[mask] + .65 * color
        y, x = np.where(mask)
        if len(x):
            point = (int(np.median(x)), int(np.median(y)))
            cv2.putText(canvas, str(index), point, cv2.FONT_HERSHEY_SIMPLEX,
                        .38, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, str(index), point, cv2.FONT_HERSHEY_SIMPLEX,
                        .38, (255, 255, 255), 1, cv2.LINE_AA)
    if removed is not None:
        removed = removed.astype(bool)
        canvas[removed] = (.15 * canvas[removed] + .85 * np.array([0, 0, 255])).astype(np.float32)
    if added is not None:
        added = added.astype(bool)
        canvas[added] = (.15 * canvas[added] + .85 * np.array([0, 200, 0])).astype(np.float32)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def titled(title, image):
    result = cv2.copyMakeBorder(image, 28, 0, 0, 0, cv2.BORDER_CONSTANT,
                                value=(255, 255, 255))
    cv2.putText(result, title, (5, 19), cv2.FONT_HERSHEY_SIMPLEX,
                .48, (0, 0, 160), 1, cv2.LINE_AA)
    return result


def connect_supported_components(mask, target_allowed, prior, args):
    """Bridge short gaps only inside the joint target/reference corridor."""
    result = mask.astype(bool).copy()
    actions = []
    prior_corridor = cv2.dilate(prior.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
        (2 * args.reference_connection_dilation + 1,) * 2)).astype(bool)
    corridor = target_allowed & prior_corridor
    count, labels, stats, _ = cv2.connectedComponentsWithStats(result.astype(np.uint8), 8)
    components = [(label, int(stats[label, cv2.CC_STAT_AREA]))
                  for label in range(1, count)
                  if int(stats[label, cv2.CC_STAT_AREA]) >= args.substantial_pixels]
    if len(components) <= 1:
        return result, actions
    main_label = max(components, key=lambda item: item[1])[0]
    connected = labels == main_label
    for label, area in sorted((item for item in components if item[0] != main_label),
                              key=lambda item: item[1], reverse=True):
        component = labels == label
        connected_yx = np.column_stack(np.where(connected))
        component_yx = np.column_stack(np.where(component))
        tree = cKDTree(connected_yx)
        distances, indices = tree.query(component_yx, k=1)
        choice = int(np.argmin(distances)); distance = float(distances[choice])
        if distance > args.connection_max_gap:
            actions.append(dict(action='keep_gap_unconnected', component=label,
                                distance=distance, reason='gap_too_large'))
            connected |= component
            continue
        start_yx = connected_yx[int(indices[choice])]
        end_yx = component_yx[choice]
        line = np.zeros_like(result, np.uint8)
        cv2.line(line, (int(start_yx[1]), int(start_yx[0])),
                 (int(end_yx[1]), int(end_yx[0])), 1, 1)
        line_mask = line.astype(bool)
        support = float((line_mask & corridor).sum() / max(1, line_mask.sum()))
        if support < args.connection_corridor_support:
            actions.append(dict(action='keep_gap_unconnected', component=label,
                                distance=distance, corridor_support=support,
                                reason='insufficient_target_reference_corridor'))
            connected |= component
            continue
        bridge = cv2.dilate(line, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
        bridge &= corridor
        proposed = result | bridge
        added = int(np.count_nonzero(proposed & ~result))
        limit = max(args.tiny_pixels, round(result.sum() * args.max_bridge_fraction))
        if added <= limit and component_count(proposed, args.substantial_pixels) < component_count(result, args.substantial_pixels):
            result = proposed
            connected |= component | bridge
            actions.append(dict(action='connect_target_reference_corridor', component=label,
                                distance=distance, corridor_support=support, added_pixels=added))
        else:
            connected |= component
            actions.append(dict(action='keep_gap_unconnected', component=label,
                                distance=distance, corridor_support=support,
                                added_pixels=added, reason='bridge_did_not_pass_limits'))
    return result, actions


def remove_edge_spurs(mask, prior, args):
    """Prune tiny weakly connected edge pixels outside the reference support."""
    result = mask.astype(bool).copy(); removed_total = 0; actions = []
    protected = cv2.dilate(prior.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
    kernel = np.ones((3, 3), np.uint8)
    for iteration in range(args.spur_iterations):
        neighbors = cv2.filter2D(result.astype(np.uint8), cv2.CV_16S, kernel,
                                 borderType=cv2.BORDER_CONSTANT) - result.astype(np.int16)
        candidates = result & (neighbors <= args.spur_max_neighbors) & ~protected
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), 8)
        removable = np.zeros_like(result)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area <= args.spur_pixels:
                removable |= labels == label
        if not removable.any():
            break
        proposed = result & ~removable
        if not proposed.any() or component_count(proposed, args.substantial_pixels) > component_count(result, args.substantial_pixels):
            break
        removed = int(removable.sum()); result = proposed; removed_total += removed
        actions.append(dict(action='remove_edge_spurs', iteration=iteration + 1,
                            removed_pixels=removed))
    return result, actions


def smooth_boundary(mask, target_allowed, prior, args):
    """Close short boundary gaps and remove small protrusions with bounded change."""
    result = mask.astype(bool)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (args.smooth_close_kernel,) * 2)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (args.smooth_open_kernel,) * 2)
    candidate = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_CLOSE,
                                 close_kernel) > 0
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN,
                                 open_kernel) > 0
    prior_allowed = cv2.dilate(prior.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    additions = candidate & ~result & target_allowed & prior_allowed
    removals = result & ~candidate
    proposed = (result & ~removals) | additions
    changed = int(additions.sum() + removals.sum())
    limit = max(args.tiny_pixels, round(result.sum() * args.max_smoothing_change_fraction))
    if (proposed.any() and changed <= limit
            and component_count(proposed, args.substantial_pixels)
            <= component_count(result, args.substantial_pixels)):
        return proposed, [dict(action='smooth_boundary',
                               close_kernel=args.smooth_close_kernel,
                               open_kernel=args.smooth_open_kernel,
                               added_pixels=int(additions.sum()),
                               removed_pixels=int(removals.sum()))]
    return result, []


def fill_internal_holes(mask, maximum_area):
    """Fill enclosed background components up to maximum_area pixels."""
    result = mask.astype(bool).copy()
    background = (~result).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background, 8)
    filled = 0
    for label in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[label]]
        touches_border = x == 0 or y == 0 or x + width == mask.shape[1] or y + height == mask.shape[0]
        if not touches_border and area <= maximum_area:
            result[labels == label] = True
            filled += area
    return result, ([dict(action='fill_internal_holes', added_pixels=filled)] if filled else [])


def keep_largest_component(mask, minimum=1):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 2:
        return mask.astype(bool), []
    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)]
    largest = 1 + int(np.argmax(areas))
    result = labels == largest
    removed = int(mask.sum() - result.sum())
    return result, [dict(action='force_keep_largest_component',
                         components_before=count - 1, removed_pixels=removed,
                         largest_area=int(result.sum()), minimum_component_area=minimum)]


def clean_one(masks, priors, target, probabilities, args):
    masks = masks.astype(bool)
    original = masks.copy()
    allowed = cv2.dilate(target.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                         (2 * args.target_dilation + 1,) * 2)).astype(bool)
    masks &= allowed[None]
    union_other = [np.any(np.delete(masks, i, axis=0), axis=0)
                   if len(masks) > 1 else np.zeros_like(target) for i in range(len(masks))]
    actions = []
    cleaned = []
    for index, mask in enumerate(masks):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        require(count > 1, 'empty mask after target constraint: %d' % index)
        areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)]
        main_label = 1 + int(np.argmax(areas))
        main = labels == main_label
        main_area = int(main.sum())
        prior = priors[index] > .5
        prior_dilated = cv2.dilate(prior.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
        main_support = float((main & prior_dilated).sum() / main_area)
        result = main.copy()
        stroke_actions = []
        for label in range(1, count):
            if label == main_label:
                stroke_actions.append(dict(component=label, area=main_area, action='keep_main',
                                           prior_support=main_support))
                continue
            component = labels == label
            area = int(component.sum())
            relative = float(area / main_area)
            redundancy = float((component & union_other[index]).sum() / area)
            own_support = float((component & prior_dilated).sum() / area)
            other_supports = [float((component & (cv2.dilate((priors[j] > .5).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0)).sum() / area)
                              if j != index else -1.0 for j in range(len(masks))]
            best_other = int(np.argmax(other_supports)) if len(masks) > 1 else -1
            best_other_support = other_supports[best_other] if best_other >= 0 else -1.0
            own_probability = None
            best_other_probability = None
            probability_margin = None
            if probabilities is not None:
                own_probability = float(probabilities[index][component].mean())
                other_values = [float(probabilities[j][component].mean())
                                if j != index else -1.0 for j in range(len(masks))]
                best_other_probability = max(other_values)
                probability_margin = own_probability - best_other_probability
            tiny_limit = max(args.tiny_pixels, round(mask.sum() * args.tiny_fraction))
            reason = None
            if area <= tiny_limit:
                reason = 'remove_tiny'
            elif (main_support >= args.minimum_main_prior_support
                  and relative <= args.foreign_relative
                  and redundancy >= args.foreign_overlap
                  and (probability_margin is None or probability_margin <= args.probability_margin)):
                reason = 'remove_redundant_foreign_component'
            elif (relative <= .5 and redundancy >= .5
                  and best_other_support >= .5
                  and best_other_support - own_support >= args.prior_margin
                  and (probability_margin is None or probability_margin <= args.probability_margin)):
                reason = 'remove_stronger_other_prior'
            if reason is None:
                result |= component
                reason = 'keep_ambiguous'
            stroke_actions.append(dict(component=label, area=area, relative_to_main=relative,
                redundancy_with_other_masks=redundancy, own_prior_support=own_support,
                best_other_stroke=best_other, best_other_prior_support=best_other_support,
                own_probability=own_probability, best_other_probability=best_other_probability,
                probability_margin=probability_margin, action=reason))

        result, connection_actions = connect_supported_components(
            result, allowed, prior, args)
        stroke_actions.extend(connection_actions)

        # Fallback for subpixel/very short gaps supported by both target and prior.
        if component_count(result, args.substantial_pixels) > 1:
            for kernel_size in (3, 5):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (kernel_size, kernel_size))
                closed = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
                proposed = result | (closed & allowed & prior_dilated)
                added = int(np.count_nonzero(proposed & ~result))
                if (component_count(proposed, args.substantial_pixels)
                        < component_count(result, args.substantial_pixels)
                        and added <= max(args.tiny_pixels,
                                         round(result.sum() * args.max_bridge_fraction))):
                    result = proposed
                    stroke_actions.append(dict(action='connect_supported_gap',
                                               kernel_size=kernel_size, added_pixels=added))
                    break
        result, spur_actions = remove_edge_spurs(result, prior, args)
        stroke_actions.extend(spur_actions)
        result, smoothing_actions = smooth_boundary(result, allowed, prior, args)
        stroke_actions.extend(smoothing_actions)
        result, hole_actions = fill_internal_holes(result, args.max_hole_area)
        stroke_actions.extend(hole_actions)
        if not args.allow_multiple_components:
            result, largest_actions = keep_largest_component(result)
            stroke_actions.extend(largest_actions)
        cleaned.append(result)
        actions.append(dict(stroke_index=index, main_area=main_area,
                            main_prior_support=main_support,
                            components_before=count - 1,
                            components_after=component_count(result), actions=stroke_actions))
    return original, np.asarray(cleaned, bool), actions


def metrics(masks, target):
    union = np.any(masks, axis=0)
    return dict(iou=iou(union, target), recall=float((union & target).sum() / target.sum()),
                precision=float((union & target).sum() / union.sum()),
                missed_pixels=int((target & ~union).sum()), extra_pixels=int((union & ~target).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    p.add_argument('--whole-iou', type=float, default=.87)
    p.add_argument('--tiny-pixels', type=int, default=16); p.add_argument('--tiny-fraction', type=float, default=.005)
    p.add_argument('--substantial-pixels', type=int, default=16)
    p.add_argument('--foreign-relative', type=float, default=.25); p.add_argument('--foreign-overlap', type=float, default=.60)
    p.add_argument('--minimum-main-prior-support', type=float, default=.30)
    p.add_argument('--prior-margin', type=float, default=.20); p.add_argument('--probability-margin', type=float, default=-.05)
    p.add_argument('--target-dilation', type=int, default=1); p.add_argument('--max-bridge-fraction', type=float, default=.03)
    p.add_argument('--connection-max-gap', type=float, default=8.0)
    p.add_argument('--connection-corridor-support', type=float, default=.80)
    p.add_argument('--reference-connection-dilation', type=int, default=4)
    p.add_argument('--spur-pixels', type=int, default=12)
    p.add_argument('--spur-max-neighbors', type=int, default=2)
    p.add_argument('--spur-iterations', type=int, default=2)
    p.add_argument('--smooth-close-kernel', type=int, default=5)
    p.add_argument('--smooth-open-kernel', type=int, default=3)
    p.add_argument('--max-smoothing-change-fraction', type=float, default=.05)
    p.add_argument('--max-hole-area', type=int, default=256)
    p.add_argument('--allow-multiple-components', action='store_true',
                   help='Keep unresolved components; default forces one largest component per mask')
    args = p.parse_args()
    out = output_dir(args.output); rows = records(args.manifest); result=[]; reports=[]; views=[]; errors=[]
    for row in rows:
        try:
            z = arrays(row['extracted_path']); masks=z['extractnet_strokes'].astype(bool)
            priors=z['sdnet_transformed_single_strokes']; probabilities=z.get('extractnet_probabilities')
            target=gray(row['image_path'])<128
            original, cleaned, actions=clean_one(masks,priors,target,probabilities,args)
            before, after=metrics(original,target),metrics(cleaned,target)
            destination=out/row['sample_id'];destination.mkdir()
            npz=destination/'cleaned.npz'
            np.savez_compressed(npz,extractnet_strokes=cleaned.astype(np.uint8),
                                stroke_labels=z['stroke_labels'].astype(np.int64),
                                source_extracted_sha256=np.array(digest(row['extracted_path'])))
            image=gray(row['image_path']);removed=original & ~cleaned;added=cleaned & ~original
            comparison=np.hstack([titled('before',overlay(image,original)),
                                  titled('after',overlay(image,cleaned)),
                                  titled('red removed / green added',
                                         overlay(image,cleaned,removed.any(0),added.any(0)))])
            comparison_path=destination/'comparison.png';cv2.imwrite(str(comparison_path),comparison)
            substantial=[component_count(mask,args.substantial_pixels) for mask in cleaned]
            status=('below_whole_iou' if after['iou']<args.whole_iou else
                    'needs_review_fragments' if any(value>1 for value in substantial) else 'clean_candidate')
            report=dict(sample_id=row['sample_id'],character=row['character'],before=before,after=after,
                        removed_pixels=int(removed.sum()),added_pixels=int(added.sum()),cleanup_status=status,
                        substantial_components=substantial,strokes=actions,
                        probabilities_available=probabilities is not None,
                        comparison_path=str(comparison_path),cleaned_path=str(npz))
            write_json(destination/'cleanup.json',report);reports.append(report)
            item=dict(row,raw_extracted_path=row['extracted_path'],extracted_path=str(npz),
                      cleanup_status=status,cleanup_report_path=str(destination/'cleanup.json'))
            result.append(item);views.append(dict(sample_id=row['sample_id'],character=row['character'],
                                                   cleanup_status=status,overlay_path=str(comparison_path)))
        except Exception as exc:
            errors.append(dict(sample_id=row.get('sample_id'),error=str(exc)))
    write_records(out/'manifest.jsonl',result);write_records(out/'errors.jsonl',errors)
    write_json(out/'report.json',dict(input_samples=len(rows),processed=len(result),errors=len(errors),
        whole_iou_threshold=args.whole_iou,
        status_counts={status:sum(r['cleanup_status']==status for r in reports)
                       for status in ('clean_candidate','needs_review_fragments','below_whole_iou')},
        removed_pixels=sum(r['removed_pixels'] for r in reports),
        added_pixels=sum(r['added_pixels'] for r in reports),samples=reports))
    preview(out,views,'Mask cleanup: before / after / removed pixels')
    if errors: raise SystemExit(1)


if __name__=='__main__': main()
