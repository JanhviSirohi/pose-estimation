import argparse
import cv2
import common
from models import coco
from pose import get_pose_data, compare


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--template', help='Path to the goal pose image.')
    parser.add_argument('--frame', help='Path to the input image.')
    parser.add_argument('--proto', help='Path to the .prototxt file')
    parser.add_argument('--model', help='Path to the .caffemodel file')
    parser.add_argument('--dataset', choices=['COCO', 'MPI'], help='Specify what kind of model was trained. It could be (COCO, MPI) depends on the dataset.')
    parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')

    args = parser.parse_args()

    frame = cv2.imread(args.frame)
    template = cv2.imread(args.template)

    if args.dataset == 'COCO':
        body_parts, pose_pairs, pair_colors = dataset_info = (coco.BODY_PARTS, coco.POSE_PAIRS, coco.PAIR_COLORS)
    else:
        body_parts, pose_pairs, pair_colors = dataset_info = (mpi.BODY_PARTS, mpi.POSE_PAIRS, mpi.PAIR_COLORS)

    network = cv2.dnn.readNetFromCaffe(common.find_file(args.proto), common.find_file(args.model))

    frame_points, frame_vectors = get_pose_data(frame, args.thr, network, dataset_info)
    template_points, template_vectors = get_pose_data(template, args.thr, network, dataset_info)

    comparison_result = compare(frame_vectors, template_vectors)

    print("   Frame Points: \n{}".format(frame_points))
    print("Template Points: \n{}".format(template_points))

    print("   Frame Vectors: \n{}".format(frame_vectors))
    print("Template Vectors: \n{}".format(template_vectors))

    print("Comparison Result: {}".format([(r, p) for r, p in zip(comparison_result, pose_pairs)]))

    print("  Color Reference: {}".format([(p, c) for p, c in zip(pose_pairs, pair_colors)]))

    common.draw_vectors(frame_points, pose_pairs, body_parts, pair_colors, frame)
    common.draw_vectors(template_points, pose_pairs, body_parts, pair_colors, template)

    cv2.imshow('Frame', frame)
    cv2.imshow('Template', template)

    cv2.waitKey()

main()