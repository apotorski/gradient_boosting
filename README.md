# Gradient Boosting
Implementation of the gradient boosting algorithm with perfect binary decision trees as weak learners.

## Derivation
The algorithm sequentially (tree by tree) builds a decision forest through optimization of the loss function - weighted sum of differentiable per-sample loss functions and regularization term
```math
L(\theta + \Delta\theta, y, w) = \sum_{i \in I} w_{i} l(\theta_{i} + \Delta\theta, y_{i}) + \frac{\lambda}{2}\|\Delta\theta\|_{2}^{2}
```
where $\theta$ are sums of already existing leaf weights associated with a given learning example, $\Delta\theta$ is the currently learned leaf weight, $y$ are labels, $w$ are weights, $I$ is a set of learning example indexes associated with a learned leaf weight, $l$ is a per-sample loss function that measures a quality of prediction and $\lambda$ is a regularization coefficient. This loss function can be approximated with the second-order Taylor expansion
```math
L(\theta + \Delta\theta, y, w) \approx \sum_{i \in I} w_{i} \left( l(\theta_{i}, y_{i}) + \frac{\partial l}{\partial \theta}(\theta_{i}, y_{i})\Delta\theta + \frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i})\frac{\Delta\theta^{2}}{2} \right) + \frac{\lambda}{2}\|\Delta\theta\|_{2}^{2}
```
Derivatives of the loss function's approximation equal
```math
\frac{\partial L}{\partial \Delta\theta}(\theta + \Delta\theta, y, w) \approx \sum_{i \in I} w_{i} \left( \frac{\partial l}{\partial \theta}(\theta_{i}, y_{i}) + \frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i})\Delta\theta \right) + \lambda\Delta\theta
```
```math
\frac{\partial^{2} L}{\partial \Delta\theta^{2}}(\theta + \Delta\theta, y, w) \approx \sum_{i \in I} w_{i} \frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda
```
Therefore, if
```math
\sum_{i \in I} w_{i} \frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda \geq 0
```
then the approximate optimal value of a leaf weight $\Delta\theta^{*}$ fulfills the following equation
```math
\sum_{i \in I} w_{i} \left( \frac{\partial l}{\partial \theta}(\theta_{i}, y_{i}) + \frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i})\Delta\theta^{*} \right) + \lambda\Delta\theta^{*} = 0
```
and the approximate optimal leaf weight equals
```math
\boxed{ \Delta\theta^{*} = -\frac{\sum_{i \in I} w_{i}\frac{\partial l}{\partial \theta}(\theta_{i}, y_{i})}{\sum_{i \in I} w_{i}\frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda} }
```

The split selection requires a method of split evaluation. The approximation of the loss function can be rewritten as follows
```math
L(\theta + \Delta\theta, y, w) \approx \sum_{i \in I} w_{i} l(\theta_{i}, y_{i}) + \Delta\theta \sum_{i \in I} w_{i} \frac{\partial l}{\partial \theta}(\theta_{i}, y_{i}) + \frac{\Delta\theta^{2}}{2} \left( \sum_{i \in I} w_{i} \frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda \right)
```
After substituting a leaf weight $\Delta\theta^{*}$
```math
L(\theta + \Delta\theta^{*}, y, w) \approx \sum_{i \in I} w_{i} l(\theta_{i}, y_{i}) - \frac{\sum_{i \in I} w_{i}\frac{\partial l}{\partial \theta}(\theta_{i}, y_{i})}{\sum_{i \in I} w_{i}\frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda} \sum_{i \in I} w_{i} \frac{\partial l}{\partial \theta}(\theta_{i}, y_{i}) + \frac{1}{2} \left(-\frac{\sum_{i \in I} w_{i}\frac{\partial l}{\partial \theta}(\theta_{i}, y_{i})}{\sum_{i \in I} w_{i}\frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda} \right)^{2} \left( \sum_{i \in I} w_{i} \frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda \right)
```
and simplifying terms, the loss function's approximation given that $\Delta\theta^{*}$ is used as a leaf weight, equals to
```math
L(\theta + \Delta\theta^{*}, y, w) \approx \sum_{i \in I} w_{i} l(\theta_{i}, y_{i}) - \frac{1}{2} \frac{\left( \sum_{i \in I} w_{i}\frac{\partial l}{\partial \theta}(\theta_{i}, y_{i}) \right)^{2}}{\sum_{i \in I} w_{i}\frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda}
```
A loss associated with a split (a pair of feature index $j$ and threshold $t$) is a sum of losses associated with generated learning example subsets
```math
\sum_{i \in I_{\text{L}}} w_{i} l(\theta_{i}, y_{i}) - \frac{1}{2} \frac{\left( \sum_{i \in I_{\text{L}}} w_{i}\frac{\partial l}{\partial \theta}(\theta_{i}, y_{i}) \right)^{2}}{\sum_{i \in I_{\text{L}}} w_{i}\frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda} + \sum_{i \in I_{\text{R}}} w_{i} l(\theta_{i}, y_{i}) - \frac{1}{2} \frac{\left( \sum_{i \in I_{\text{R}}} w_{i}\frac{\partial l}{\partial \theta}(\theta_{i}, y_{i}) \right)^{2}}{\sum_{i \in I_{\text{R}}} w_{i}\frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda}
```
where $x$ are features, $I_{\text{L}} = \{ i| x_{i, j} \leq t \}$ and $I_{\text{R}} = \{ i| x_{i, j} > t \}$. This expression can be simplified (because $I_{\text{L}} \cup I_{\text{R}} = I$)
```math
\sum_{i \in I} w_{i} l(\theta_{i}, y_{i}) - \frac{1}{2} \left( \frac{\left( \sum_{i \in I_{\text{L}}} w_{i}\frac{\partial l}{\partial \theta}(\theta_{i}, y_{i}) \right)^{2}}{\sum_{i \in I_{\text{L}}} w_{i}\frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda} + \frac{\left( \sum_{i \in I_{\text{R}}} w_{i}\frac{\partial l}{\partial \theta}(\theta_{i}, y_{i}) \right)^{2}}{\sum_{i \in I_{\text{R}}} w_{i}\frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda} \right)
```
After removing values that don't depend on the choice of split, the criterion equals
```math
-\left( \frac{\left( \sum_{i \in I_{\text{L}}} w_{i}\frac{\partial l}{\partial \theta}(\theta_{i}, y_{i}) \right)^{2}}{\sum_{i \in I_{\text{L}}} w_{i}\frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda} + \frac{\left( \sum_{i \in I_{\text{R}}} w_{i}\frac{\partial l}{\partial \theta}(\theta_{i}, y_{i}) \right)^{2}}{\sum_{i \in I_{\text{R}}} w_{i}\frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda} \right)
```
The optimal split minimizes the derived criterion
```math
j^{*}, t^{*} = \arg\min_{j, t} -\left( \frac{\left( \sum_{i \in I_{\text{L}}} w_{i}\frac{\partial l}{\partial \theta}(\theta_{i}, y_{i}) \right)^{2}}{\sum_{i \in I_{\text{L}}} w_{i}\frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda} + \frac{\left( \sum_{i \in I_{\text{R}}} w_{i}\frac{\partial l}{\partial \theta}(\theta_{i}, y_{i}) \right)^{2}}{\sum_{i \in I_{\text{R}}} w_{i}\frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda} \right)
```
```math
\boxed{ j^{*}, t^{*} = \arg\max_{j, t} \left( \frac{\left( \sum_{i \in I_{\text{L}}} w_{i}\frac{\partial l}{\partial \theta}(\theta_{i}, y_{i}) \right)^{2}}{\sum_{i \in I_{\text{L}}} w_{i}\frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda} + \frac{\left( \sum_{i \in I_{\text{R}}} w_{i}\frac{\partial l}{\partial \theta}(\theta_{i}, y_{i}) \right)^{2}}{\sum_{i \in I_{\text{R}}} w_{i}\frac{\partial^{2} l}{\partial \theta^{2}}(\theta_{i}, y_{i}) + \lambda} \right) }
```
Splits are generated sequentially (level by level) until the desired height is reached.

## Quick start
To install dependencies, run
```bash
pip install -r requirements.txt
```

To train a decision forest, run
```bash
python src/train_forest.py \
    --iteration_number 1000 \
    --height 6 \
    --regularization_coefficient 1.0 \
    --leaf_weight_update_number 10 \
    --learning_rate 0.5 \
    --bin_number 256 \
    --test_size 0.2 \
    --validation_size 0.2
```

## References
[Chen, T. and Guestrin, C., 2016. XGBoost: A Scalable Tree Boosting System. arXiv preprint arXiv:1603.02754.](https://arxiv.org/abs/1603.02754)
