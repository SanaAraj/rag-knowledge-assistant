"""Generate clean, multi-page ML reference PDFs for the RAG demo.

Each top-level section starts on its own page so that source attribution
(document name + page number) is crisp and verifiable in the demo.
"""
import sys
from fpdf import FPDF

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."


def make_pdf(path, title, sections):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_title(title)
    for i, (heading, body) in enumerate(sections):
        pdf.add_page()
        if i == 0:
            pdf.set_font("Helvetica", "B", 20)
            pdf.multi_cell(0, 11, title)
            pdf.ln(4)
        pdf.set_font("Helvetica", "B", 15)
        pdf.multi_cell(0, 9, heading)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 11.5)
        for para in body.strip().split("\n\n"):
            pdf.multi_cell(0, 6.5, para.strip())
            pdf.ln(3)
    pdf.output(path)
    print("wrote", path)


intro = [
    ("What Is Machine Learning?",
     """Machine learning is a subfield of artificial intelligence in which systems learn patterns from data rather than following explicitly programmed rules. Instead of a human writing out every decision, the system is shown many examples and adjusts an internal model so that its predictions improve over time.

A machine learning workflow usually has four stages: collecting and preparing data, choosing a model, training the model on the data, and evaluating how well it generalizes to examples it has never seen. The three broad families of machine learning are supervised learning, unsupervised learning, and reinforcement learning."""),

    ("Supervised Learning",
     """Supervised learning trains a model on labeled examples, where every input is paired with the correct output. The goal is to learn a mapping from inputs to outputs that generalizes to new, unseen inputs. Because the correct answers are provided during training, the model can measure its error directly and correct itself.

Supervised learning splits into two main task types. Classification predicts a discrete category, such as whether an email is spam or not spam. Regression predicts a continuous value, such as the price of a house. Common supervised algorithms include linear and logistic regression, decision trees, support vector machines, and neural networks."""),

    ("Unsupervised Learning",
     """Unsupervised learning works with data that has no labels. Rather than predicting a known answer, the model tries to discover structure that is already present in the data. This is useful when labeling is expensive or when the interesting patterns are not known in advance.

Two common unsupervised tasks are clustering and dimensionality reduction. Clustering groups similar examples together; k-means is a widely used clustering algorithm. Dimensionality reduction, such as principal component analysis, compresses data into fewer dimensions while preserving as much information as possible, which helps with visualization and with removing noise."""),

    ("Reinforcement Learning",
     """Reinforcement learning trains an agent to make a sequence of decisions by interacting with an environment. The agent takes actions, receives rewards or penalties, and adjusts its policy to maximize the total reward it collects over time. Unlike supervised learning, it is not told the correct action; it must discover which actions pay off through trial and error.

Reinforcement learning is a natural fit for problems such as game playing, robotics, and recommendation, where decisions made now affect opportunities later. A central challenge is the trade-off between exploration, trying new actions to gather information, and exploitation, using what is already known to earn reward."""),

    ("Overfitting and Generalization",
     """A model overfits when it memorizes the training data, including its noise, instead of learning the underlying pattern. An overfit model scores very well on data it has already seen but performs poorly on new data, which defeats the purpose of learning.

To detect overfitting, data is split into a training set and a held-out test set; a large gap between training accuracy and test accuracy is a warning sign. Common remedies include gathering more data, simplifying the model, applying regularization to penalize complexity, and using cross-validation to estimate performance more reliably."""),
]

nn = [
    ("Neural Networks: The Basics",
     """A neural network is a model made of layers of simple units called neurons, loosely inspired by biological brains. Each neuron computes a weighted sum of its inputs, adds a bias, and passes the result through a non-linear activation function. Stacking many neurons lets the network represent complex relationships that a single linear model cannot.

A typical feedforward network has an input layer, one or more hidden layers, and an output layer. The values that the network learns are the weights and biases; training is the process of adjusting them so the network's outputs match the desired targets."""),

    ("Layers and Activation Functions",
     """Activation functions give a neural network its ability to model non-linear patterns. Without them, stacking layers would be equivalent to a single linear transformation. The rectified linear unit, or ReLU, outputs zero for negative inputs and the input itself for positive inputs, and it is the most common choice in hidden layers because it is simple and trains quickly.

Other activations include the sigmoid, which squashes values into the range zero to one and is useful for probabilities, and softmax, which turns a vector of scores into a probability distribution across classes at the output layer."""),

    ("Training with Backpropagation",
     """Neural networks are trained with backpropagation combined with gradient descent. First, a forward pass computes the network's prediction and a loss function measures how far that prediction is from the target. Backpropagation then applies the chain rule of calculus to compute the gradient of the loss with respect to every weight in the network.

Gradient descent uses those gradients to nudge each weight in the direction that reduces the loss, scaled by a learning rate. Repeating this forward-and-backward cycle over many batches of data gradually lowers the loss. Choosing a good learning rate matters: too large and training diverges, too small and it crawls."""),

    ("Deep Learning Applications",
     """Deep learning refers to neural networks with many hidden layers, which can learn hierarchical features directly from raw data. Convolutional neural networks excel at images by detecting edges, then shapes, then objects. Recurrent networks and transformers handle sequences such as text and speech.

These architectures power everyday applications including image recognition, machine translation, speech-to-text, and recommendation systems. Their main costs are the need for large datasets and significant compute for training, though a trained model can often run efficiently afterward."""),
]

make_pdf(f"{OUT_DIR}/introduction-to-machine-learning.pdf",
         "Introduction to Machine Learning", intro)
make_pdf(f"{OUT_DIR}/neural-networks-and-deep-learning.pdf",
         "Neural Networks and Deep Learning", nn)
