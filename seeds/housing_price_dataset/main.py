def main(train_dataset, test_dataset, metric_fn, loss_fn, device):
    model = Model()
    model_paramters_count = count_parameters(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    metric = 0
    batch_count = 0

    for epoch in range(epochs):
        for X, y in train_dataset:
            X = X.to(device)
            y = torch.unsqueeze(y, 1).to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        for X, y in test_dataset:
            X = X.to(device)
            y = torch.unsqueeze(y, 1).to(device)
            output = model(X)
            metric += metric_fn(y, output)
            batch_count += 1

    metric /= batch_count

    return float(metric), int(model_paramters_count)