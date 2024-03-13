import os, logging, shutil, json
import torch
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import CommonMetricPrinter, EventStorage, JSONWriter, TensorboardXWriter
import detectron2.utils.comm as comm
from detectron2.data import build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.structures.instances import Instances

logger = logging.getLogger("detectron2")

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='train faster rcnn')
    parser.add_argument('--config-file', default="", metavar='FILE', help='path to config file');
    parser.add_argument('--imgs-dir', default="", help='path to images');
    parser.add_argument('--annos-file', default="", help='path to annotations file');
    parser.add_argument('--resume', action="store_true", help='resume training');
    args = parser.parse_args()
    return args


def do_train(cfg, model, resume = False, val_data_name=None):
    
    model.train()
    print(model)
    
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer = optimizer, scheduler = scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume = resume).get("iteration", -1) + 1
    )
    if resume is False:
        start_iter = 0;
    
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter = max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )
    
    data_loader = build_detection_train_loader(cfg)
    
    logger.info("Starting training from iteration {} to iteration {} with saving period {}".format(start_iter, max_iter, cfg.SOLVER.CHECKPOINT_PERIOD))
    
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()
            
            loss_dict = model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if iteration - start_iter > 5 and (iteration % 100 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def main():
    args = parse_args();
    print("\nCommand Line Args:", args);
    print("\n")

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file);

    register_coco_instances(cfg.DATASETS.TRAIN[0], {}, args.annos_file, args.imgs_dir);
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True);

    if args.resume is False:
        # copy config (*.yaml) to OUTPUT_DIR for reference
        shutil.copy(args.config_file, cfg.OUTPUT_DIR);

        # save arguments to OUTPUT_DIR for reference
        with open(cfg.OUTPUT_DIR + "/args.txt", 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    model = build_model(cfg);
    do_train(cfg, model, resume=args.resume);
    print("Done!!!");

if __name__ == "__main__":
    main();