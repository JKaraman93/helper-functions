## Custom loop with Tqdm
```
n_steps = 100
with trange(1, n_epochs + 1, desc="All epochs") as epochs:
    for epoch in epochs:
        mean_loss.reset_state()
        for metric in metrics:
            metric.reset_state()
        with trange(1, n_steps + 1, desc=f"Epoch {epoch}/{n_epochs}") as steps:
            for step in steps:
                X_batch, y_batch = random_batch(X_train, y_train)
                with tf.GradientTape() as tape:
                    y_pred = model(X_batch)
                    main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                    loss = tf.add_n([main_loss] + model.losses)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                for variable in model.variables:
                    if variable.constraint is not None:
                        variable.assign(variable.constraint(variable))                    
                status = OrderedDict()
                mean_loss(loss)
                status["loss"] = mean_loss.result().numpy()
                for metric in metrics:
                    metric(y_batch, y_pred)
                    status[metric.name] = metric.result().numpy()
                if step==steps.total:
                  y_pred = model(X_valid)
                  status["val_loss"] = np.mean(loss_fn(y_valid, y_pred))
                  status["val_accuracy"] = np.mean(tf.keras.metrics.sparse_categorical_accuracy(
                      tf.constant(y_valid, dtype=np.float32), y_pred))
                steps.set_postfix(status)


```
