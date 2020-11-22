from torchvision import transforms
import torch

class Preprocess:
    def preprocess_image_data(self, input):
        # sample execution (requires torchvision)
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        output = []
        for i in range(len(input)):
            input_tensor = preprocess(input[i])
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
            output.append(input_batch)
        result = torch.cat(output, dim=0) 

        return result