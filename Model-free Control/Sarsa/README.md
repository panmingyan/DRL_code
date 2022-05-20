# Sarsa

## 使用说明

运行```task0_train.py```即可

## 环境说明

使用openai gym中的[Cliff Walking](https://www.gymlibrary.ml/environments/toy_text/cliff_walking/)

## 算法伪代码

![sarsa_algo](assets/sarsa_algo.png)

## 其他说明

### 与Q-learning区别

算法上区别很小，只在更新公式上，但Q-learning是Off-policy，而Sarsa是On-policy，可参考[知乎：强化学习中sarsa算法是不是比q-learning算法收敛速度更慢？](https://www.zhihu.com/question/268461866)
