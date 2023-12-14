def main(train_dataset, test_dataset, metric_fn, loss_fn, device):
    model = Model()
    model_paramters_count = count_parameters(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    metric = 0
    samples_count = 0

    for epoch in range(epochs):
        for X, y in train_dataset:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        for X, y in test_dataset:
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            predicted_classes = torch.argmax(output, dim=1)
            metric += metric_fn(y, predicted_classes)
            samples_count += len(y)

    metric = metric / samples_count

    return float(metric), int(model_paramters_count)
