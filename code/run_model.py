
from src.dataloader import AugmentorChain


if __name__ == '__main__':

    ac = AugmentorChain(
        random_seed=1,
        augmentors=[
            FlipAugmentor(),
            RotateAugmentor(),
            PixelNoiseAugmentor(scale=0.1),
            ChannelNoiseAugmentor(scale=0.1)
        ]
    )

    rsdata_train = RSData(
        './data/combined.zarr',
        mask_area=[1],
        cutout_size=21,
        augmentor_chain=ac
    )

    rsdata_valid = RSData(
        './data/combined.zarr',
        mask_area=[2],
        cutout_size=21,
        augmentor_chain=ac,
        rs_means=rsdata_train.rs_means,
        rs_stds=rsdata_train.rs_stds
    )

    print('Creating dataloaders...')

    train_dl = DataLoader(rsdata_train, batch_size=32, shuffle=True, num_workers=6)
    valid_dl = DataLoader(rsdata_valid, batch_size=32, shuffle=False, num_workers=6)

    print('Creating dataloaders done.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    num_classes = 10
    num_epochs = 2
    # batch_size = 16
    learning_rate = 0.01

    # Model
    model = ResNet(inplanes=4, num_classes=num_classes).to(device)
    print('Number of trainable parameters: ', count_trainable_parameters(model))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.000, momentum=0.9)

    total_step = len(train_dl)

    for epoch in range(num_epochs):
        for i, (images, labels) in tqdm(
            enumerate(train_dl),
            total=total_step,
            desc=f'Epoch {epoch+1}/{num_epochs} (train)'
        ):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_dl:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))
