=> What we've established:
1. shown that circuits are preserved across clients, as well as within client for IID case.
2. shown that circuits are not preserved across clients in extreme label skew case. 
3. for uniform no of classes for each client in extreme label skew, the local mask global model accuracy remains ~100%, showing that the same circuit is valid for global model as well.   

=> Potential directions:
1. in general, analyse how one class circuit dominates other class circuits in global modle, and what happens if we remove that dominating circuit (do the other classes perform well now?) 
2. we can do neuron patching in a non-iid toy example
3. we can train SAE on our client models' circuits, and on a resnet model circuit to see if they encode the same features
4. after fedavg, are the neurons becoming polysemantic? (how are their semanticity being effected, are they encoding the same features?) We can take an image, and make a heatmap on it to show if this is true or not
5. show that after fedavging, we get bloated circuits (if it took 10 neurons to identify the task, it is now taking 20 neurons to do the same thing)
6. okay so let's say class 9 has really low accuracy on global validation set. we check which circuit lights up the most on all images of 9. this most frequently activated circuit can be the class that is dominating



abdullah  

- Non iid: rounds 1class each, 2class each, 3class each (to show uniformness) 
- common class example label skew: 2class sab mei diff, 1 class same   
- ablation: we can do neuron patching in a non-iid toy example: 
- circuit cascade (non iid circuits ko ensamble and see performance, and that should beat fedavg in non iid) -> head ko avg? 