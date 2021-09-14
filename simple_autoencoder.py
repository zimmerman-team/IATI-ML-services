import torch
import pymongo
import pickle
import numpy as np

MONGODB_CONN="mongodb://mongouser:XGkS1wDyb4922@localhost:27017/learning_sets"

class AE(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = torch.nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = torch.nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = torch.nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = torch.nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

def main():
    client = pymongo.MongoClient(MONGODB_CONN)
    db = client['learning_sets']
    coll = db['npas_tsets']
    document = coll.find({'rel':'budget'})[0]
    train_dataset = pickle.loads(document['train_npa']).astype(np.float32)
    test_dataset = pickle.loads(document['test_npa']).astype(np.float32)

    print("train_dataset.shape",train_dataset.shape)
    input_cardinality = train_dataset.shape[1]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(input_shape=input_cardinality).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = torch.nn.MSELoss()

    epochs = 1000
    for epoch in range(epochs):
        loss = 0
        for batch_features in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, input_cardinality).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

if __name__ == "__main__":
    main()