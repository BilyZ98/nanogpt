import torch
import torch.profiler as profiler

# Example training loop with profiling
with profiler.profile(
    schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Measure data loading time manually if desired
            inputs, targets = batch

            # Synchronize before starting GPU timing
            torch.cuda.synchronize()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Optionally, measure specific operation time with CUDA events here
            
            # Step the profiler
            prof.step()
