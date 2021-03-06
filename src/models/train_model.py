import os
import subprocess

import hydra
import matplotlib.pyplot as plt
import model as md
import torch
from model import Network
from torch import nn, optim

import src.data.get_dataset

from datetime import datetime


# wandb.init(project="mlops-project", entity="fuzzy-fish-waffle")


@hydra.main(config_name="training_conf.yaml", config_path="../../conf")
def main(cfg):
    """Training loop"""

    print("Training day and night")
    model = Network(cfg.hyperparameters.num_classes)
    # Magic
    # wandb.watch(model, log_freq=cfg.print_every)
    trainloader, _, testloader = src.data.get_dataset.main(cfg)

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.hyperparameters.lr,
        momentum=cfg.hyperparameters.momentum,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, mode="max", verbose=True
    )

    minibatch_loss_list = []
    steps = 0
    running_loss = 0
    losses = []
    timestamp = []
    epochs = cfg.hyperparameters.epochs
    print_every = cfg.hyperparameters.print_every

    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1

            optimizer.zero_grad()

            labels = labels.type(torch.LongTensor)
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            minibatch_loss_list.append(loss.item())

            if steps % print_every == 0:
                # wandb.log({"loss": loss})

                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = md.validation(model, testloader, criterion)

                print(
                    "Epoch: {}/{}.. ".format(e + 1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                    "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
                    "Test Accuracy: {:.3f}".format(accuracy / len(testloader)),
                )

                losses.append(running_loss / print_every)
                timestamp.append(steps)
                running_loss = 0
                # Make sure dropout and grads are on for training
                model.train()
        scheduler.step(minibatch_loss_list[-1])

    model.eval()

    plt.plot(timestamp, losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.savefig(hydra.utils.get_original_cwd() + "/reports/figures/training.png")

    # Quantization
    # model_int8 = torch.quantization.convert(model)

    # plt.show()
    checkpoint = {
        "state_dict": model.state_dict(),
    }

    date_time = datetime.now().strftime("%m%d%Y%H%M%S")

    script_model = torch.jit.script(model)
    script_model.save(
        "{cwd}/models/deployable_model_{date_time}.pth".format(
            cwd=hydra.utils.get_original_cwd(), date_time=date_time
        )
    )

    torch.save(
        checkpoint,
        "{cwd}/models/checkpoint_{date_time}.pth".format(
            cwd=hydra.utils.get_original_cwd(), date_time=date_time
        ),
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
        subprocess.check_call(
            [
                "gsutil",
                "cp",
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    f"models/deployable_model_{date_time}.pth",
                ),
                os.path.join(cfg.cloud.path_deploy, f"deployable_model_{date_time}.pt"),
            ]
        )


if __name__ == "__main__":
    main()
