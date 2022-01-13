from model import Network
import model as md
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import hydra
import os
from src.data.make_dataset import FishDataset
#import wandb


@hydra.main(config_name= "training_conf.yaml" ,config_path="../../conf")
def main(cfg):
    """Training loop
    """
    os.chdir(hydra.utils.get_original_cwd())
    print("Working directory : {}".format(os.getcwd()))
    print("Training day and night")    
    model = Network(cfg.num_classes)   


    # Magic
    # wandb.watch(model, log_freq=cfg.print_every)

    model.train()
    trainloader = torch.load(cfg.train_data)
    testloader = torch.load(cfg.test_data)

    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       factor=0.1,
                                                       mode='max',
                                                       verbose=True)


    minibatch_loss_list = []
    steps = 0
    running_loss = 0
    losses = []
    timestamp = []
    epochs = cfg.epochs
    print_every = cfg.print_every
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


                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

                losses.append(running_loss / print_every)
                timestamp.append(steps)
                running_loss = 0
                # Make sure dropout and grads are on for training
                model.train()
        scheduler.step(minibatch_loss_list[-1])
    plt.plot(timestamp, losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.savefig("reports/figures/training.png")

    #plt.show()
    checkpoint = {
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, "models/checkpoint.pth")




if __name__ == "__main__":
    main()
