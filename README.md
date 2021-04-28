# Non-robust Features #

<b>Adversarial Vulnerability and Transferability : Non-Robust Features</b>

Here, we will learn the nature of Non-robust features in context of adversarial vulnerability and adversarial transferability. By following the Ilyas definition of Robust and Non-robust features,which is as follow:

A feature f is useful for a dataset D when there exists a ρ > 0 such that,

<img src="https://render.githubusercontent.com/render/math?math=\mathrm{E}_{(x,y) \in \mathbf{D}}[y.f(x)] \ge p">

A feature f is robust when, for some γ > 0,

<img src="https://render.githubusercontent.com/render/math?math=\mathrm{E}_{(x,y) \in \mathbf{D}}[\min_{\|\delta\|_2 \le \varepsilon}y.f(x+\delta)] \ge \gamma">

A feature is said to be Non-robust when it is useful but not Robust.

The robustified test sets are build based on the definition of robust features to show that on the generated robust test sets the non-robust classifier achieves substantially higher accuracy which indicates that the classifier must be using non-robust features that are entangled with robust features. To generate robustified test set,compute a set of adversarial examples for each batch during training.

<img src="https://render.githubusercontent.com/render/math?math=\theta^* = \argmin_{\theta}\mathrm{E}_(x,y \in \mathbf{D})[\max_{\|\delta\|_2 \le \varepsilon}\mathcal{L}(\theta,x+\delta,y)]">

In short, only robust features are useful for classification. Therefore while validating(original CIFAR test set as validation set), by using the output of the m-dimensional representation layer of robust classifier,the set of images with robust features are generated.

First install the library mentioned in <b>required_library.txt</b> file before execution. Here, the <b>robustness</b> package by <b>madry lab</b> is used to create robustifeid test sets.

<b>Evaluation:</b>

<b>Dataset:</b> CIFAR-10 (10,000 images as test set)

<b>Model:</b> ResNet50

<b>Attack:</b> PGD (L2 norm)

<b>Robustified test sets:</b> R0(epsilon(PGD): 0),R0.125(epsilon(PGD): 0.125),R0.5(epsilon(PGD): 0.5),R1(epsilon(PGD): 1)

<b>PGD parameters:</b> Epsilon=0.5, step-size=0.05, step=10

<b>Batch-size:</b> 200

<b>Epoch:</b> 100

<b>Optimizer:</b> Adam with learning rate=0.001

<b>Scheduler:</b> MultiStepLR with milestones=[100, 150]