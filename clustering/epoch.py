>>> for epoch in range(num_epochs):
... for x_batch, y_batch in train_dl:
... # 1. Generate predictions
... pred = model(x_batch)[:, 0]
... # 2. Calculate loss
... loss = loss_fn(pred, y_batch)
... # 3. Compute gradients
... loss.backward()
... # 4. Update parameters using gradients
... optimizer.step()
... # 5. Reset the gradients to zero
... optimizer.zero_grad() 
... if epoch % log_epochs==0:
... print(f'Epoch {epoch} Loss {loss.item():.4f}')