import logging
import numpy as np

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.events import EventStorage
from detectron2.data import build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import transforms as T
from detectron2.data import DatasetCatalog
from detectron2.data.samplers import InferenceSampler
from detectron2.data import DatasetMapper  

from WSTTA.wstta import WSTTA
from WSTTA.utils import create_weak_label

logger = logging.getLogger("detectron2");


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='train faster rcnn')
    parser.add_argument('--config-file', default="", metavar='FILE', help='path to config file');
    parser.add_argument('--imgs-dir', default="", help='path to images');
    parser.add_argument('--annos-file', default="", help='path to annotations file');
    parser.add_argument('--num-adapt', type=int, default=-1, help='number of adaptation samples');
    parser.add_argument('--mom-init', type=float, default=0.1, help="initial momentum");
    parser.add_argument('--mom-lb', type=float, default=0.0, help="lower bound of momentum");
    parser.add_argument('--omega', type=float, default=1.0, help="decay factor of momemtum");
    parser.add_argument('--alpha', type=float, default=0.1, help="alpha value for image level loss");
    parser.add_argument('--psd-thr', type=float, default=0.8, help="prob threshold to create pseudo labels");
    args = parser.parse_args()
    return args

def forward_and_adapt(args, cfg, model):
    model.eval();

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS);
    optimizer = build_optimizer(cfg, model)
    
    hl_tta = WSTTA(args, model=model, optimizer=optimizer);
    
    total_samples = len(DatasetCatalog.get(cfg.DATASETS.TRAIN[0]))
    
    data_loader = build_detection_train_loader(cfg, 
                                              mapper=DatasetMapper(cfg, is_train=True, augmentations=[T.ResizeShortestEdge(short_edge_length=(0,0), max_size=1333, sample_style='choice')]),
                                              sampler=InferenceSampler(total_samples));
    
    evaluator = COCOEvaluator(cfg.DATASETS.TRAIN[0], cfg, False, output_dir="./output/");
    evaluator.reset()
    
    print("\n\n\n========== Perform HL-TTA with %d samples ======================" % (args.num_adapt));
    datasets = data_loader.dataset.dataset.dataset;
    np.random.seed(9999);
    indices = np.random.permutation(total_samples);
    
    with EventStorage(0) as storage:
        for idx, sample_i in enumerate(indices):
            inputs = [datasets[sample_i]];
            
            # if the current sample < num_adapt, then we perform prediction then adaptation
            # otherwise we only perform prediction
            if idx < args.num_adapt:
                inputs_weak = create_weak_label(inputs);  # provide weak labels
                outputs = hl_tta.forward_then_adapt(inputs_weak);
                status = "Forward and adapt sample"
                    
            else:
                outputs = hl_tta.forward_only(inputs);
                status = "Forward sample"
            
            evaluator.process(inputs, outputs)
            
            if idx % 500 == 0:
                print("%s %d/%d" % (status, idx, total_samples));
            if idx == args.num_adapt - 1:
                print("\t===> Finish adapting model at sample %d" % (idx));
                
    results = evaluator.evaluate();
    return results;


def main():
    args = parse_args();
    print("\nCommand Line Args:", args);
    print("\n")

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file);

    register_coco_instances(cfg.DATASETS.TRAIN[0], {}, args.annos_file, args.imgs_dir);
    
    model = build_model(cfg);
    forward_and_adapt(args, cfg, model);
    print("Done!!!");

if __name__ == "__main__":
    main();