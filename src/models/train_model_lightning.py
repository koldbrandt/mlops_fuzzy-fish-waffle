import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from model_lightning import LightningModel
from src.data import get_dataset
import hydra
from datetime import datetime
import torchdrift
import copy
import subprocess
import os
import sklearn.manifold
from matplotlib import pyplot


@hydra.main(config_name="training_conf.yaml", config_path="../../conf")
def main(cfg):

    trainloader, _, testloader = get_dataset.main(cfg)

    model = LightningModel(10)

    wd_logger = loggers.WandbLogger(
        name="test", entity="fuzzy-fish-waffle", project="mlops-project"
    )
    trainer = pl.Trainer(logger=wd_logger, max_epochs=cfg.hyperparameters.epochs)

    trainer.fit(model, trainloader, testloader)

    # plt.show()
    # checkpoint = {
    #     "state_dict": model.state_dict(),
    # }

    date_time = datetime.now().strftime("%m%d%Y%H%M%S")

    trainer.save_checkpoint(
        "{cwd}/models/checkpoint_{date_time}.pth".format(
            cwd=hydra.utils.get_original_cwd(), date_time=date_time
        )
    )

    if cfg.cloud.save:
        subprocess.check_call(
            [
                "gsutil",
                "cp",
                os.path.join(
                    hydra.utils.get_original_cwd(), f"models/checkpoint_{date_time}.pth"
                ),
                os.path.join(cfg.cloud.path, f"model_{date_time}.pt"),
            ]
        )
    # torch.save(
    #     checkpoint,
    #     "{cwd}/models/checkpoint_{date_time}.pth".format(
    #         cwd=hydra.utils.get_original_cwd(), date_time=date_time
    #     ),
    # )

    inputs, _ = next(iter(trainloader))
    inputs_ood = corruption_function(inputs)

    # N = 6
    model.eval()
    # inps = torch.cat([inputs[:N], inputs_ood[:N]])
    model.cpu()
    # predictions = model.predict(inps).max(1).indices

    feature_extractor = copy.deepcopy(model)
    feature_extractor.classifier = torch.nn.Identity()

    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()

    torchdrift.utils.fit(trainloader, feature_extractor, drift_detector)

    features = feature_extractor(inputs)
    score = drift_detector(features)
    p_val = drift_detector.compute_p_value(features)
    print(f"score: {score}, p_val: {p_val}")

    # N_base = drift_detector.base_outputs.size(0)
    mapper = sklearn.manifold.Isomap(n_components=2)
    base_embedded = mapper.fit_transform(drift_detector.base_outputs)
    features_embedded = mapper.transform(features.detach().numpy())
    pyplot.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c="r")
    pyplot.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    pyplot.title(f"score {score:.2f} p-value {p_val:.2f}")
    print("saving the plot")
    pyplot.savefig("drift_detection.png")

    features = feature_extractor(inputs_ood)
    score = drift_detector(features)
    p_val = drift_detector.compute_p_value(features)
    print(f"score_ood: {score}, p_val_ood: {p_val}")

    features_embedded = mapper.transform(features.detach().numpy())
    pyplot.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c="r")
    pyplot.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    pyplot.title(f"score {score:.2f} p-value {p_val:.2f}")
    pyplot.savefig("drift_detection_ood.png")


def corruption_function(x: torch.Tensor):
    return torchdrift.data.functional.gaussian_blur(x, severity=2)


if __name__ == "__main__":
    main()
