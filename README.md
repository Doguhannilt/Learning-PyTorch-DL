<p><strong>Acknowledgment: </strong> I would like to express my gratitude to Mr. Daniel Bourke (mrdbourke) for his exceptional PyTorch tutorial.
Mr. Bourke's comprehensive and well-structured tutorial on PyTorch has been an invaluable resource in my learning journey. His deep understanding of PyTorch and his ability to explain intricate concepts in a clear and concise manner have greatly contributed to my understanding of this powerful deep learning framework.
I am grateful for Mr. Bourke's dedication to providing high-quality educational content, which has played a crucial role in enhancing my skills and knowledge in PyTorch.
</p>
<div align ="center">
  <strong>Github Repository: </stong>https://www.youtube.com/watch?v=Z_ikDlimN6A&t=28813s 
  </strong>
</div>

<br>
<div align="center">
  <img src="https://user-images.githubusercontent.com/42931974/68615320-d6bf3380-04e8-11ea-84f8-dcef049f1ed3.gif" alt="alternatetext" height=300 weight = 300>
</div>
<h1>PyTorch vs TensorFlow</h1>

<lu>
	<li>They differ because PyTorch has a more "pythonic" approach and is object-oriented, while TensorFlow offers a variety of options. PyTorch is used for many deep learning projects today, and its popularity is increasing among AI researchers, although of the three main frameworks, it is the least popular.</li>
</lu>

<br></br>


<italic>PyTorch is an open-source machine learning library for Python, developed primarily by Facebook's AI Research lab. It provides a flexible and efficient framework for building and training deep learning models. PyTorch is known for its dynamic computation graph, which allows developers to define and modify models on-the-fly, making it particularly suitable for research and prototyping.</italic>
<br>


<h3>Why</h3> 
PyTorch? PyTorch has gained popularity due to its intuitive and Pythonic syntax, which makes it easier for researchers and developers to understand and work with. It provides extensive support for neural networks and deep learning techniques and offers a rich ecosystem of tools and libraries.




<h3>Who uses PyTorch?</h3> 
PyTorch is used by researchers, data scientists, and machine learning practitioners across academia and industry. Many companies and research institutions utilize PyTorch for various applications, including computer vision, natural language processing, reinforcement learning, and more.




<h3>What is PyTorch?</h3> 
PyTorch is a Python library that provides a set of high-level functions and classes for building and training deep learning models. It includes tensor computation, automatic differentiation, and tools for data loading and preprocessing. PyTorch also supports GPU acceleration, enabling efficient computation on graphics processing units.




<h3>Definition and Detail: </h3> 
PyTorch is based on the Torch library, originally developed for Lua programming language. It was created to address the limitations of static computation graphs by introducing a dynamic computation graph. This graph allows for easy model construction, dynamic control flow, and debugging.
PyTorch's core component is its multi-dimensional array, called a "tensor," which is similar to NumPy arrays but with additional features and compatibility with GPUs. PyTorch tensors support various mathematical operations and can be used for storing and manipulating numerical data.
Another key feature of PyTorch is automatic differentiation, which enables the computation of gradients automatically. This functionality is crucial for training deep learning models using gradient-based optimization algorithms such as stochastic gradient descent (SGD). PyTorch's autograd module provides automatic differentiation capabilities, allowing users to define and compute gradients without manual calculations.
Overall, PyTorch's flexibility, ease of use, and extensive community support have made it a popular choice among researchers and practitioners in the field of deep learning.

<br></br>


<h1>ARCHITECTURE OF PYTORCH</h1>

<br>
<lu>
	<li>Import the necessary modules:</li>
  <img src="https://github.com/Doguhannilt/Zomato-Analysis-With-Python/assets/77373443/8d128bd3-70bf-425f-bffe-a6278241814d">

  <li>Define the architecture of your model by subclassing nn.Module:</li>
Define the layers and operations in the __init__ method.<br>
Implement the forward pass in the forward method.
Instantiate an instance of your model.<br>
	<li>Define the loss function:</li>
Choose from various loss functions provided by torch.nn.
	<li>Define the optimizer:</li>
Choose from various optimization algorithms provided by torch.optim.
	<li>Prepare your data:</li>
Load and preprocess your dataset using PyTorch's data loading utilities or custom data loaders.<br>
	<li>Train your model:</li>
Iterate over your dataset, forward pass the inputs through the model, calculate the loss, and perform backpropagation to update the model's parameters.
	<li>Evaluate your model:</li>
Use a separate validation dataset or test dataset to evaluate the performance of your trained model.
	<li>Save and load your model:</li>
Save the trained model's parameters to disk for future use or load pre-trained models.
<br>
</lu>

<h1>OPTIMIZERS</h1>
<lu>
	<li>Stochastic Gradient Descent (SGD): A classic optimization algorithm that updates model parameters by computing gradients on a small subset of the training data at each iteration.</li><br>
	<li>Adam: An adaptive optimization algorithm that combines the advantages of adaptive learning rates and momentum. It adapts the learning rate for each parameter based on the past gradients and updates.</li><br>
	<li>Adagrad: An optimizer that adapts the learning rate for each parameter based on the historical gradients. It gives larger updates to infrequent parameters and smaller updates to frequent ones.</li><br>
	<li>Adadelta: Similar to Adagrad, Adadelta is an adaptive learning rate optimization algorithm. It improves upon Adagrad by addressing its learning rate decay over time.</li><br>
	<li>Adamax: An extension of Adam optimizer that incorporates the infinity norm (max norm) of the gradients. It is particularly useful when dealing with sparse gradients.</li><br>
	<li>RMSprop: An optimization algorithm that uses a moving average of squared gradients to adapt the learning rate for each parameter. It helps mitigate the diminishing learning rate problem in Adagrad.</li><br>
	<li>SparseAdam: An Adam variant specifically designed for sparse gradients, where only the non-zero gradient elements are updated.</li><br>
	<li>AdamW: An extension of Adam optimizer that incorporates weight decay (L2 regularization) directly into the update step. It provides better handling of weight decay compared to vanilla Adam.</li><br>
	<li>ASGD (Averaged Stochastic Gradient Descent): An optimization algorithm that maintains a running average of model parameters during training. It can be useful for large-scale distributed training.</li><br>
	<li>LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno): A quasi-Newton optimization algorithm that approximates the inverse Hessian matrix using limited memory. It is typically used for small-to-medium-sized problems due to its memory requirements.</li><br>
	<li>Rprop (Resilient Backpropagation): An optimization algorithm that updates model parameters based on the sign of the gradient. It adapts the learning rate per parameter and handles learning rate adjustments more robustly than gradient-based methods.</li><br>
</lu>
<div align="center">
  <img src="https://i.stack.imgur.com/gjDzm.gif" weight =300 height =300>
  <p><strong>Credit: </strong> https://stats.stackexchange.com/questions/357449/two-large-decreses-in-loss-function-with-adam-optimizer</p>
</div>
<br>
<br>
<h1>LOSS</h1>

<lu>
	<li>Mean Squared Error (MSE): Calculates the mean squared difference between predicted and target values. It is commonly used for regression problems.</li><br>
	<li>Binary Cross Entropy (BCE): Computes the binary cross-entropy loss between predicted probabilities and binary targets. It is often used for binary classification tasks.</li><br>
	<li>Cross Entropy Loss: Measures the dissimilarity between predicted and target probability distributions. It is commonly used for multi-class classification problems, often in combination with softmax activation.</li><br>
	<li>L1 Loss (Mean Absolute Error): Computes the mean absolute difference between predicted and target values. It is robust to outliers and commonly used for regression tasks.</li><br>
	<li>Smooth L1 Loss (Huber Loss): A combination of L1 and L2 losses, which provides a smooth transition between the two. It is often used in object detection tasks to handle bounding box regression.</li><br>
	<li>Kullback-Leibler Divergence (KL Divergence): Measures the difference between two probability distributions. It is commonly used in tasks such as variational autoencoders and generative models.</li><br>
	<li>Cosine Similarity Loss: Computes the cosine similarity between predicted and target vectors. It is often used in tasks such as recommendation systems and similarity-based learning.</li><br>
	<li>Hinge Loss: Used for maximum-margin classification problems, such as support vector machines (SVMs). It encourages correct classification while penalizing margin violations.</li><br>
	<li>Triplet Loss: Used in tasks such as face recognition and metric learning, where the goal is to learn embeddings that maximize the similarity between similar samples and minimize the similarity between dissimilar samples.</li><br>
</lu>

<div align ="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/1*47skUygd3tWf3yB9A10QHg.gif" weight = 300 height =300>
</div>
  
<br>
<br>
<h1>SIMPLE TRAIN-TEST LOOP</h1>


<div align="center">
	<img src ="https://github.com/Doguhannilt/Zomato-Analysis-With-Python/assets/77373443/b5592ec7-c690-4896-8363-e9e857d0f4fc">
	<br>
	<img src="https://github.com/Doguhannilt/Zomato-Analysis-With-Python/assets/77373443/dfa1a3af-97c3-42d4-8e0e-c0e129f001be">
</div>

<br>
<br>
<h1>CLASSIFICATION AND LINEAR REGRESSION MODELS BY USING NN.SEQUENTIAL</h1>

<div align="center">
	<img src="https://github.com/Doguhannilt/Zomato-Analysis-With-Python/assets/77373443/ce7e8607-5d58-4ca3-abf2-c66f3c67f559">
</div>
<br>
<br>
<h1>THE LIST OF ACTIVATION FUNCTIONS IN DEEP LEARNING (WITH DEFINITION)</h1>
<lu>
	<li>ReLU (Rectified Linear Unit): An activation function that sets negative input values to zero and keeps positive values unchanged.</li><br>
	<li>Sigmoid: An activation function that maps input values to the range [0, 1], commonly used for binary classification tasks.</li><br>
	<li>Tanh (Hyperbolic Tangent): An activation function that maps input values to the range [-1, 1], preserving the sign of the input.</li><br>
	<li>LeakyReLU: A variation of ReLU that introduces a small slope for negative input values, allowing a small gradient and preventing zero outputs for negative inputs.</li><br>
	<li>ELU (Exponential Linear Unit): An activation function that smoothly handles negative input values and provides robustness to noisy data.</li><br>
	<li>SELU (Scaled Exponential Linear Unit): A variation of ELU that helps normalize the activations and maintain a mean of zero and unit variance during training, improving the performance of deep neural networks.</li><br>
	<li>Softmax: An activation function that normalizes the input values into a probability distribution over multiple classes.</li><br>
	<li>LogSoftmax: A logarithmic transformation of the softmax function that helps stabilize the computation and is often used in conjunction with the negative log-likelihood loss for classification tasks.</li><br>

</lu>
<div align="center">
<img src="https://miro.medium.com/v2/resize:fit:640/1*xYCVODGB7RwJ9RynebB2qw.gif" height =300 weight =300>
</div>

<br>
<br>
<h1>WHAT IS LEARNING RATE?</h1>

<lu>
	<li>Definition: The learning rate is a hyperparameter that determines the step size at which the model's parameters are updated during training.</li><br>
	<li>Description: It controls the trade-off between convergence speed and optimization stability. A higher learning rate allows for faster convergence but risks overshooting the optimal solution, while a lower learning rate leads to slower convergence but potentially more accurate results.</li><br>
</lu><br>
<div align="center">
  <img src="https://gbhat.com/assets/gifs/sgd_learning_rates.gif" weight=300 height=300>
  <p><strong>Credit: </strong>https://gbhat.com/machine_learning/gradient_descent_learning_rates.html</p>
</div>

<br>
<h1>BETTER VISUALIZATION TECHNIQUES</h1>
<div align="center">
	<h3>Plot Decision Boundary</h3>
	<img src="https://github.com/Doguhannilt/Zomato-Analysis-With-Python/assets/77373443/2833dfe0-3e25-4dbb-bb34-d54ced9875a0"><br>
	<h3>Plot Linear Data or Training-Test-Predictions</h3>
	<img src="https://github.com/Doguhannilt/Zomato-Analysis-With-Python/assets/77373443/984f51e3-d688-49fc-9291-6f906118cdb9"><br>
</div>

<br>
<br>
<h1>HOW TO IMPROVE A MODEL?</h1>
<lu>
	<li>Add more layers</li><br>
	<li>Add more hidden units</li><br>
	<li>Fitting for longer</li><br>
	<li>Changing the activation functions</li><br>
	<li>Change the learning rate</li><br>
	<li>Change the loss function</li><br>
</lu>
<br>


<h1>WHICH LOSS FUNCTION AND OPTIMIZER THAT WE SHOULD USE?</h1>
<div align="center">
	<img src = "https://github.com/Doguhannilt/Zomato-Analysis-With-Python/assets/77373443/e7a0558f-58a5-4839-ba43-ffb49b843856">
	<p>Credit: https://www.learnpytorch.io/01_pytorch_workflow/</p>
<br>
<br>
<h1>UNDERSTANDING THE MATH OF DEEP LEARNING</h1>

<h3>Forward Propagation:</h3>
Forward propagation is the process of passing input data through a neural network to obtain the output. Here's a simplified overview:<br>


<lu>
	<li>Input: The neural network takes input data, which can be a vector or a matrix.</li><br>
	<li>Hidden Layers: The input data is passed through multiple layers of the neural network. Each layer consists of weights and biases, which are adjusted during training.</li><br>
	<li>Activation Function: After calculating the weighted sum of inputs and applying biases, an activation function is applied to introduce non-linearity into the network.</li><br>
	<li>Output: The final layer produces the output of the neural network, which could be a single value or a vector representing different classes or values.</li><br>
</lu>

<br>
<h3>Backpropagation:</h3><br>
Backpropagation is the process of adjusting the weights and biases of a neural network based on the error calculated between the predicted output and the expected output. Here's a simplified overview:

<lu>
	<li>Error Calculation: The error is computed by comparing the predicted output with the desired output using a loss function.</li><br>
	<li>Gradient Calculation: The gradient of the loss function with respect to the weights and biases is computed to determine how they should be adjusted to minimize the error.</li><br>
	<li>Weight Update: The weights and biases of the neural network are updated in the opposite direction of the gradient, using an optimization algorithm such as stochastic gradient descent (SGD).</li><br>
	<li>Error Backpropagation: The error is then propagated back through the network, layer by layer, using the chain rule of calculus. This allows the gradients to be calculated for each layer, which helps in adjusting the weights and biases.</li><br>
</lu>

<br>

<br>

<div align="center">
	<img src ="https://7-hiddenlayers.com/wp-content/uploads/2020/06/NEURONS-IN-NUERAL-NETWORK.gif" height =300 weight = 300>
</div>

<br>
<br>

<h1>WHAT IS NON-LINEARITY?<h1>
	

<br>
<img src="https://rahsoft.com/wp-content/uploads/2021/04/Screenshot-2021-04-23-at-19.59.06.png" weight = 300 height = 300/>
<h3>CODE</h3>
<img src="https://github.com/Doguhannilt/Learning-PyTorch-DL/assets/77373443/a8b08795-4117-4732-83c2-67d0f67cfbbd" weight = 300 height = 300/>
<br>
<p><italic>Artifical Neural Networks are a large combination of linear and non-straight functions which are potentially able to find patterns in data</italic></p>

