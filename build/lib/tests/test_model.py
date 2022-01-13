# import os.path

# import pytest
# import torch

# from src.models.model import MyAwesomeModel

# @pytest.mark.skipif(not os.path.exists('data/processed/train.pt'), reason="Train files not found")

# def test_inputshape_has_outputshape():
#     train_dataset = torch.load('data/processed/train.pt')
#     BATCH_SIZE = 5
#     trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True, num_workers=2)
#     model = MyAwesomeModel() # My model

#     images, _ = next(iter(trainloader))

#     model.train()
#     # for _, data in enumerate(trainloader, 0):
#     # get the inputs, batch size of 4
#     # images, _ = data
        
#     images = images[:, None, :, :]
#     output = model(images)[0]
#     # print(output)
#     #     print(output[0].shape)

#     assert images.shape == torch.Size([BATCH_SIZE,1,28,28]) and output.shape == torch.Size([BATCH_SIZE,10]), "Shape of model input and output did not match"

# # Test raise error
# # def test_error_on_wrong_shape():
# #     with pytest.raises(ValueError, match='Expected each sample to have shape [1, 28, 28]'):
# #         model = MyAwesomeModel() # My model
# #         model(torch.randn(1,1,2,3))