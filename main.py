import os
import argparse
import json
import torch
import torch.nn.functional
import model.dinov2
import model.sam2
import model.radio
import support_util
import query_util
import metric
from chatrex.upn import UPNWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot VOC evaluation with DINOv2 + SAM2")

    parser.add_argument('--json_name_list', nargs='+', type=str)
    parser.add_argument('--feat_extractor_name',
                        type=str,
                        default='RADIO',
                        choices=['DINOV2', 'RADIO'],
                        help='feature extractor name (default: %(default)s)')

    parser.add_argument('--radio_model_path', type=str,
                        default='c-radio_v4-h',
                        help='RADIO model version when feat_extractor_name=RADIO (default: %(default)s)')

    parser.add_argument('--radio_cache_root', type=str,
                        default='./model_cache',
                        help='RADIO/model cache root (torch_hub under it) when feat_extractor_name=RADIO (default: %(default)s)')

    parser.add_argument('--sam2_model_type', type=str,
                        default='large',
                        help='SAM2 model type (small/medium/large) (default: %(default)s)')

    parser.add_argument('--dinov2_image_size', type=int,
                        default=630,
                        help='Input size for dinov2 images (default: %(default)s)')

    parser.add_argument('--data_dir', type=str,
                        default='./data/',
                        help='Root directory for dataset (default: %(default)s)')

    parser.add_argument('--test_dir', type=str,
                        default='./data/',
                        help='Root directory for exact test dataset')

    parser.add_argument('--save_dir', type=str,
                        default='./result/',
                        help='Output prediction JSON file (default: %(default)s)')

    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run models on (default: %(default)s)')
    
    parser.add_argument('--target_categories', type=str,nargs='+',
                        default=['bus','sofa','cow','bird','motorbike'],
                        help='Target categories for evaluation (default: %(default)s)')

    parser.add_argument('--min_threshold', type=float, default=0.01,
                        help='mean threshold for upn')

    parser.add_argument('--filter_by_categories', action='store_true',
                        help='filter by categories')

    parser.add_argument('--diffusion_steps', type=int, 
                        help='number of diffusion steps')

    parser.add_argument('--points_per_side', type=int,
                        default=32,
                        help='Points per side for SAM2 mask generator (default: %(default)s)')

    parser.add_argument('--alp', type=float, 
                        help='alpha in diffusion')
                        
    parser.add_argument('--lamb', type=float, 
                        help='lamda for decay')

    return parser.parse_args()


def get_support_data(test_dir, json_name_list):
    new_info_list = []
    for json_name in json_name_list:
        ori_json_path = os.path.join(test_dir, 'annotations', json_name)
        dataset_name = os.path.basename(test_dir)
        with open(ori_json_path, 'r') as rf:
            info = json.load(rf)

        img_dict, cls_dict = {}, {}
        for item in info['images']:
            img_dict[item['id']] = item['file_name']
        for item in info['categories']:
            cls_dict[item['id']] = item['name']

        new_info = {}
        for item in info['annotations']:
            save_item = {
                'image': 'CDFSOD/{}/train/{}'.format(dataset_name, img_dict[item['image_id']]),
                'bbox': item['bbox']
            }
            cls_name = cls_dict[item['category_id']]

            if cls_name in new_info:
                new_info[cls_name].append(save_item)
            else:
                new_info[cls_name] = [save_item]
        new_info_list.append(new_info)
    return new_info_list


def main():
    args = parse_args()

    print('Loading UPN...')
    ckpt_path = './checkpoints/upn_large.pth'
    upn = UPNWrapper(ckpt_path)

    print('Loading RADIO...')
    feat_extractor = torch.hub.load(
        './RADIO', 'radio_model', version=args.radio_model_path,
        source="local", progress=True, skip_validation=True, force_reload=True
    ).eval().to(args.device)
    image_transform = model.radio.RadioImageTransform(target_long_side=1260, patch_size=16)

    print('Loading SAM2...')
    sam2_model, sam2_predictor, sam2_mask_generator = model.sam2.load_sam2_components(
        model_type=args.sam2_model_type,
        device=args.device,
        points_per_side=args.points_per_side
    )
        
    # Load support set
    # json_name_list = ['1_shot.json', '5_shot.json', '10_shot.json']
    support_data_list = get_support_data(args.test_dir, args.json_name_list)

    # Build memory bank
    proto_feat_list, proto_cls_list = [], []
    for support_data in support_data_list:
        memory_bank = support_util.extract_support_features(
            support_data,
            sam2_predictor,
            args.feat_extractor_name,
            feat_extractor,
            image_transform,
            args.data_dir,
            args.device
        )
        proto_feat, proto_cls = support_util.compute_prototype_weights(memory_bank, args.device)
        proto_feat_list.append(proto_feat)
        proto_cls_list.append(proto_cls)
    # assert proto_cls_list[0] == proto_cls_list[1] == proto_cls_list[2]
    
    # Load VOC2007 test loader
    image_paths, coco_style_loader = query_util.load_voc2007_coco_json(
        os.path.join(args.test_dir, 'annotations/test.json'),
        os.path.join(args.test_dir, 'test')
    )

    # Generate predictions
    total_res_list = metric.generate_coco_style_predictions_upn(
        coco_style_loader,
        os.path.join(args.test_dir, 'test'),
        sam2_predictor,
        args.feat_extractor_name,
        feat_extractor,
        image_transform,
        proto_feat_list,
        proto_cls_list[0],
        upn,
        args.diffusion_steps,
        args.alp,
        args.lamb,
        args.device,
    )

    for _, (results, json_name) in enumerate(zip(total_res_list, args.json_name_list)):
        save_results = [
            {
                'image_id': item['image_id'],
                'category_id': item['category_id'],
                'bbox': item['bbox'],
                'score': item['score']
            } for item in results if 'negative' not in item['category_name']
        ]
        save_path = os.path.join(args.save_dir, os.path.basename(args.test_dir) + '_{}'.format(json_name.replace('_', '')))
        with open(save_path, 'w') as wf:
            json.dump(save_results, wf)


if __name__ == '__main__':
    main()

