# Screencast Notes
### This file contains notes for my screencast
##### Created by w45242hy


## Note
- 


## Content
- ### Introduction (1 minutes)
    - #### Self-introduction (10 seconds)
        - Name
        - Project title
    - #### Motivation (0.5 minutes)
        - Single model
            - Unreliable + unstable
                - Final prediction depends entirely on 1 model
        - Ensemble learning
            - Compensate for the above problem
            - Training a "committee" of models
            - Research shows that diversity improves ensemble performance
            - Aggregation also influences ensemble performance heavily
        - **In this project, we try to visualise how diversity and different aggregations impact ensemble performance**
    - #### Project architecture (0.5 minutes)
        - The roles of diversity and aggregation are examined through 5 perspectives, with each refining constraints and assumptions of the previous one (we will go through them in no time, but first, let's talk more about the mathematical theories behind)
            1. (Just showing the objective name) Develop Baseline Models
            2. (Just showing the objective name) Implement NCL Ensemble
            3. (Just showing the objective name) Integrate NEAT with NCL
            4. (Just showing the objective name) Analyse Different Aggregators
            5. (Just showing the objective name) Study Different Ensemble Topologies
- ### Main Content (6.5 minutes)
    - #### Backgroud Theory (2.5 minutes)
        - Ambiguity decomposition
            - Have a look at the paper "Neural network ensembles, cross validation, and active learning"
            - The ensemble loss is mathematically bounded by the average individual risk of base learners if MSE is used, suggesting the benefits of using an ensemble model instead of a single model
            - Do not mention proof, just explain the terms briefly
        - Bias-variance decomposition
            - Have a look at the paper "Neural networks and the bias/variance dilemma"
            - Decompose the expected risk of a model to 3 components
                - Use the example of darts
            - Do not mention proof, just explain the terms briefly
        - Bias-variance-diversity decomposition
            - Mention proof: combining Ambiguity Decomposition and Bias-variance Decomposition
            - A 4th component to control the expected ensemble risk, i.e. diversity
                - The negative sign indicates that increasing diversty reduces ensemble risk
                - Empirically measured by $\rho$
            - It is a trade-off: increasing prediction diverse also increases bias
                - How can we control the extent of diversity and visualise its effect? -> NCL
        - NCL
            - Mention its principle of forcing diversity
                - MSE for individual accuracy
                - Correlation penalty for collective diversity
    - #### Design and Implementation (2.5 minutes)
        1. Develop Baseline Models
            - ##### Summary/Motivation
                - Implement an MLP and an arithmetic-mean-aggregator
                    - Establish a performance baseline
                    - Verify the correctness of their PyTorch implementations, which will be used in future objectives
                    - Empirically prove Ambiguity Decomposition
                - Question: How much more accurate is a "committee of directors" compared to the decision of a single "chairman"? 
        2. Implement NCL Ensemble
            - ##### Summary/Motivation
                - Implement an ensemble model that uses NCL
                    - Control diversity extent and studies the impact of it to ensemble performance
                    - Empirically proves that Bias-variance-diversity Decomposition is a trade-off: moderate diversity should reduce ensemble loss but too much diversity harms ensemble performance
                - Question: How does the quality of a decision-making change when company directors are explicitly forced to have different opinions? 
        3. Integrate NEAT with NCL
            - ##### Summary/Motivation
                - Create an ensemble model that uses NEAT and NCL
                    - Figure out if architectural diversity improves/hinders ensemble performance for this regression task
                - Question: How does the quality of a decision-making change when company directors include members with different backgrounds, e.g. human + animals? 
            - ##### Background
                - Briefly descrive the general evolutionary steps
                    - Briefly explain Mutation
                    - Briefly explain Crossover
            - ##### Code Implementation
                - Take the original NEAT steps but add a step of backpropagation with NCL between reproduction and replacement
                    - Fitness of each base learner is re-calculated after backpropagation before replacement
                - $\lambda$ is dynamically adjusted instead of fixed by developers
                    - Larger value initially as diversity is little to encourage exploration
                    - Smaller value later as diversity has increased to focus more on individual accuracy
                - ##### Algorithm to Convert Dynamic Neural Network to GPU-runnable Format
                    - General steps have been detailed in report
                        - General idea: Order nodes using topological sort and a graph of fully-connected layers is created with each layer representing 1 node
        4. Analyse Different Aggregators
            - ##### Summary/Motivation
                - Study the effect of using different aggregators on ensemble performance
                    - Arithmetic-mean-aggregator
                    - Median-aggregator
                        - Take the middle value of predictions
                    - Neural-network-aggregator
                        - Assign different weightings to predictions of different base learners
                - Question: How does the quality of a decision-making change if certain company directros are given more voting power? 
        5. Study Different Ensemble Topologies
            - ##### Summary/Motivation
                - Study different ways of combining predictions from base learners, i.e. different ensemble topologies
                - Question: How does the quality of a decision-making change if company directors no longer vote in a single room but instead, lead their own specialised sub-committee? 
    - #### Evaluation (1.5 minutes)
        1. Develop Baseline Models
            - Arithmetic-mean-aggregator: Show graph of MSE against ensemble size
                - Targeted success criteria: 
                    1. Ensemble MSE should always be lower than individual MSE
                    2. Average individual MSE in the ensemble model similar to MSE of a single MLP
                    3. Support Ambiguity Decomposition
        2. Implement NCL Ensemble
            - Show graph of MSE against $\lambda$
                - Targeted success criteria: 
                    1. Initial decrease in ensemble MSE (due to increase in diversity indicated by increasing $\rho$) as weight on penalty term is increased
            - Show graph of $\rho$ against $\lambda$ to support all success criteria
                - Targeted success criteria: 
                    1. Increase in ensemble MSE (due to too much diversity indicated by a very high $\rho$) as weight on penalty term is too high
        5. Study Different Ensemble Topologies
            - Show graph of MSE of a neural-network-aggregator (if have time)
                - Target success criteria: 
                    1. Recursive topoligies give ensemble performance improvement only when a neural-network-aggregator is used because if every layer is equally important, the recursive depths become redundant
- ### Conclusion (1.5 minute)
    - ### Project contributions
        - Study the the roles of diversity and aggregation in ensemble learning through multiple mathematical theories and experiments
            - Establish a clear link between ensemble parameters, e.g. $\lambda$, and overall performance through diversity
            - Validate the benefits of ensemble learning over a single model
            - Conclude that diversity is like a double-edged sword: controlled diversity improves ensembles but too much of it hinders performance and that a balance should be strived for
        - Development of an algorithm that converts dynamic neural networks into a GPU-compatible format
        - Proposal of the recursive ensemble framework
    - ### Thank you


## Transcript
- ### Introduction
    Hi, I am Jacob. Welcome to the screencast of my final year project, where I study the roles of diversity and aggregation in ensemble learning. To understand what ensemble learning is and why we need it, let's have a look at this single AI model. It's quite big, so let's shrink it to a smaller rectangle. By feeding it some data, it returns some predictions. This is the normal pipeline of a single AI model. However, this could be unreliable as we are depending on this single AI model entirely. What if it is wrong or unstable? Ensemble learning compensates for this problem, in which this single AI model is actually a "committee" of models. And instead of training 1 model, we are actually training $m$ models with combined predictions. Research shows that increasing the diversity of these models can improve the overall performance. At the same time, the aggregation of predictions also affects the final prediction. In this project, we will try to visualise how diversity and different aggregations impact ensemble performance. By divide-and-conquer, the whole project is split into 5 sequential objectives, each refining constraints and assumptions of the previous objective. We will go through them in time, but let's further discuss the mathematical theories behind it. 
    - Around 1:15
- ### Background
    Regarding Ambiguity Decomposition, its equation here states that when squared loss is used, an arithmetic-mean ensemble performs no worse than the average performance of an individual model. Then, we have Bias-variance Decomposition. The performance of a model depends on 3 factors. We can control 2 of them, bias and variance, which can be seen as how accurate and how stable the model's prediction is respectively. Noise represents the inherent randomness that we can never reduce or remove in our prediction. In this dartboard example, the source of the noise comes from the inherent fact that the dartboard is moving and so even if you are 100% accurate and stable, all darts will never end up at the centre. Now, in ensemble learning, there is actually a 4th factor that affects the model performance, called diversity. Here, we have the expected risk to measure the performance of a single model. Now, imagine that this single model is actually an arithmetic-mean ensemble. We can then apply Ambiguity Decomposition and Bias-variance Decomposition to give 4 terms. Note that diversity term has a negative sign in front of it, suggesting that increasing diversity improves ensemble performance. But this equation actually represents a trade-off because while diversity is increasing, bias and variance are also increasing. So how can we control and prevent excessive diversity? Here comes Negative Correlation Learning. It is an ensemble technique that explicitly encourages controlled diversity. The loss function is squared loss, except a correlation penalty term is added, which penalises models for being similar. During optimisation, loss function is minimised, two things happen. The squared loss is minimised and the model becomes more accurate. At the same time, the penalty term is maximised and the model becomes more different from the rest. $\lambda$ is a non-negative term set by us that balances the above trade-off of individual accuracy and collective diversity. 
    - Around 1:50
- ### Stage
    Let's talk about the 5 project objectives. I will use a company as an analogy to explain the intuitions behind these 5 objectives. A single model can be seen as a Chairman making decision for a company. We are interested in how the company becomes better if the Chairman is replaced with a committee of directors, each representing a model itself in an ensemble model. We have implemented 2 types of models. They establish the performance baseline and determine some hyperparameter values. The models will also be used in future experiments to study other objectives so they are verified. Essentially, this objective is like a prerequisite-step for this project. Lastly, Ambiguity Decomposition is empirically proven using an experiment. For the next objective, an ensemble model that uses Negative Correlation Learning is used. Previously, a group of directors is used to make company decision. We want to know what happens if these directors are forced to have different opinions. For this objective, in addition to implementing the NCL ensemble model, we also study how diversity can be controlled by other hyperparameters and its impact on the overall performance. At this point, diversity comes from the training data and all models have the same architecture. However, for the next objective, we are interested in how diversity on a different dimension changes things, especially when all models have different structures. This is like replacing some company directors with non-human directors who have a totally different mindset and background. Furthermore, we want to determine the appropriate value of NCL $\lambda$ dynamically. NEAT seems to be the answer, but what exactly is it? This documentary will answer your questions. 
    - Around 1:35
- ### NEAT with NCL
    Somewhere in the universe, there exists a planet full of different models. A thousand years ago, they were all the same, but not anymore. This isn't just luck. It's natural selection. Just like animals, models can have their own "DNA" called genome. The models that can solve the given problem survive and evolve. Those who can't, fade away. In every evolutionary phase, there are 4 stages. During Evaluation, all models in a population are assessed. The brighter the models are, the better solutions they are. During Selection, the stronger models are selected to pass their DNA to the next generation. During Reproduction, the selected models mate and give babies using crossover and mutation. Crossover exchanges genetic code, giving the children the best traits from both parents. Mutation creates random new traits on the children to help their survival. Lastly, during Replacement, the weaker models are eliminated and the stronger ones remain. This process repeats for many thousand years. Combining with Negative Correlation Learning, 2 new evolutionary stages are added. Just like before, models on this planet are evaluated, selected and they mate. However, before removing the weaker models, the new models improve and diversify using backpropagation with NCL. The children, who are usually stronger, are then evaluated. The weaker ones are finally removed to maintain a constant population size. Evolution on this planet is a never-ending cycle. There's nothing the weaker ones can do to stop it than becoming the stronger ones. This shows the updated NCL loss function used for backpropagation. $D$ is the population diversity. When diversity is initially small, $\lambda$ is big, prioritising diversity and solution exploration. After models are diversed enough via optimisation, $\lambda$ decreases and individual accuracy is prioritised. 
    - Around 1:50
- ### Stage
    For the next objective, 3 different aggregators are used and their corresponding impacts on performance are studied. In the previous objective, company directors have the same vote count. What happens if the voting mechanism changes? We try aggregating using the mean, median and a neural network. Remember this diagram earlier? Arithmetic mean is applied. If every model has different vote counts, this resembles using a neural-network-aggregator with no hidden layers. We also try selecting the median prediction from all models. Lastly, in this objective, different ensemble topologies are studied, just like asking directors to vote in different rooms instead of a single room. In addition to proposing a recursive ensemble framework, different aggregators with ensemble topologies are studied. This is the ensemble model pipeline introduced earlier. Now, I will tell you that one of these models is actually an ensemble model itself. Let's shrink this big ensemble to a small circle where the light blue represents sub-models and the dark blue represents an aggregator. We can infinitely form many possible ensemble architectures. These are two pipelines of recursive ensemble models. 
    - Around 1:10
- ### Interesting Findings
    There are some interesting findings. First, ensemble performance improves when diversity initially increases but it deteriorates when diversity becomes excessive. The bottom graph measures diversity using diversity coefficient. After a certain point, diversity spikes, leading to a drop in ensemble performance. Furthermore, when $\lambda$ is set well, like $0.8$, simpler models are more effective. The blue line on the top graph is from an ensemble composed of simpler models. Originally, it is worse than ensembles of complex models represented by the orange and green lines. But when $\lambda$ is set well, it actually outperforms them. This suggests proper ensemble learning reduces computational resource requirements. 
    - Around 0:55
- ### Conclusion
    In conclusion, I have outlined the 5 objectives in this project and their intuitions behind the study of diversity and aggregation. I have then provided mathematical motivations and introduced NEAT with NCL. Lastly, I propose a recursive ensemble framework and outlined some interesting findings, including the encouragement of moderate diversity in ensemble learning. 
    - Around 0:20


## TODO
- [x] Add background music
    - Refer to ```Patrick Mermelstein Lyons_Dissertation-vid.mp4```
- [x] Add smooth animation