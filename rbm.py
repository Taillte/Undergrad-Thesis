from __future__ import print_function
import numpy as np

class RBM:
  
  def __init__(self, num_visible, num_hidden):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.debug_print = False

    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
    # and sqrt(6. / (num_hidden + num_visible)). One could vary the 
    # standard deviation by multiplying the interval with appropriate value.
    # Here we initialize the weights with mean 0 and standard deviation 0.1. 
    # Reference: Understanding the difficulty of training deep feedforward 
    # neural networks by Xavier Glorot and Yoshua Bengio
    np_rng = np.random.RandomState(1234)

    self.weights = np.asarray(np_rng.uniform(
			low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	size=(num_visible, num_hidden)))


    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)
    
    
  def train(self,data,epochs=100,learning_rate=0.1,batch_size=100,k=3):
    num_samples = data.shape[0]
    num_batches = int(num_samples/batch_size)
    errors = np.zeros(epochs)
    for epoch in range(epochs):
        if epoch > epochs*(0.2):
            learning_rate = learning_rate*0.1
            if epoch > epochs*(0.5):
                learning_rate = learning_rate*0.1
        for batch in range(num_batches):
            mini_data = data[batch*batch_size:((batch+1)*batch_size-1),:]
            errors[epoch]+= self.train_mini_batches(mini_data, learning_rate, k)
        errors[epoch]=errors[epoch]/num_batches
    print('errors are ',errors)
    return errors
            
            

  def train_mini_batches(self, data, learning_rate, k):
    """
    Train the machine.

    Parameters
    ----------
    data: A matrix where each row is a training example consisting of the states of visible units.    
    """

    num_examples = data.shape[0]

    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)

    # Clamp to the data and sample from the hidden units.
      
    pos_hidden_activations = np.dot(data, self.weights)
    pos_hidden_probs = self._logistic(pos_hidden_activations)
    pos_hidden_probs[:,0] = 1 # Fix the bias unit.
    pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
    # themselves, when computing associations. We could also use the states; see section 3 of Hinton's
    # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
    pos_associations = np.dot(data.T, pos_hidden_probs)

    # Reconstruct the visible units and sample again from the hidden units
    # negative CD phase
    neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
    neg_visible_probs = self._logistic(neg_visible_activations)
    neg_visible_probs[:,0] = 1 # Fix the bias unit.
    neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
    neg_hidden_probs = self._logistic(neg_hidden_activations)
    neg_hidden_states = neg_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Note, again, that we're using the activation *probabilities* when computing associations, not the states
    # themselves.
    neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
      
    for i in range(int(k-1)):
        neg_visible_activations = np.dot(neg_hidden_states, self.weights.T)
        neg_visible_probs = self._logistic(neg_visible_activations)
        neg_visible_probs[:,0] = 1 # Fix the bias unit.
        neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
        neg_hidden_probs = self._logistic(neg_hidden_activations)
        neg_hidden_states = neg_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
        # Note, again, that we're using the activation *probabilities* when computing associations, not the states
        # themselves.
        neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

    # Update weights.
    self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)
    error = np.mean(np.abs((pos_associations - neg_associations) / num_examples))
      
    return error

  def run_visible(self, data):
    
    num_examples = data.shape[0]
    
    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))
    
    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = self._logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Always fix the bias unit to 1.
    # hidden_states[:,0] = 1
  
    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states
    
  # TODO: Remove the code duplication between this method and `run_visible`?
  def run_hidden(self, data):

    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the visible units.
    visible_activations = np.dot(data, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    visible_probs = self._logistic(visible_activations)
    # Turn the visible units on with their specified probabilities.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    # Always fix the bias unit to 1.
    # visible_states[:,0] = 1

    # Ignore the bias units.
    visible_states = visible_states[:,1:]
    return visible_states
    
  def daydream(self, num_samples, initialising_data):
    """
    Initialize the visible units once, and start running alternating Gibbs sampling steps
    (where each step consists of updating all the hidden units, and then updating all of the visible units),
    taking a sample of the visible units at each step.
    Note that we only initialize the network *once*, so these samples are correlated.

    Returns
    -------
    samples: A matrix, where each row is a sample of the visible units produced while the network was
    daydreaming.
    """

    # Create a matrix, where each row is to be a sample of of the visible units 
    # (with an extra bias unit), initialized to all ones.
    samples = np.ones((num_samples, self.num_visible + 1))

    # If taking the first sample from a uniform distribution.
    # samples[0,1:] = np.random.rand(self.num_visible)
    samples[0,1:] = initialising_data

    # Start the alternating Gibbs sampling.
    # Note that we keep the hidden units binary states, but leave the
    # visible units as real probabilities. See section 3 of Hinton's
    # "A Practical Guide to Training Restricted Boltzmann Machines"
    # for more on why.
    for i in range(1, num_samples):
      visible = samples[i-1,:]

      # Calculate the activations of the hidden units.
      hidden_activations = np.dot(visible, self.weights)      
      # Calculate the probabilities of turning the hidden units on.
      hidden_probs = self._logistic(hidden_activations)
      # Turn the hidden units on with their specified probabilities.
      hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
      # Always fix the bias unit to 1.
      hidden_states[0] = 1

      # Recalculate the probabilities that the visible units are on.
      visible_activations = np.dot(hidden_states, self.weights.T)
      visible_probs = self._logistic(visible_activations)
      visible_states = visible_probs > np.random.rand(self.num_visible + 1)
      samples[i,:] = visible_states

    # Ignore the bias units (the first column), since they're always set to 1.
    return samples[:,1:]        
      
  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))




# Stacking two RBMS

class Stacked_RBM:

    def __init__(self, num_visible, num_first_hidden, num_second_hidden):
        self.first_rbm = RBM(num_visible = num_visible, num_hidden = num_first_hidden)
        self.second_rbm = RBM(num_first_hidden,num_second_hidden)
        
    def train(self,data,epochs=100,learning_rate=0.1,batch_size=100,k=3):
    
        self.first_rbm.train(data,epochs=epochs,learning_rate=learning_rate,batch_size=batch_size,k=k)
        
        # Want to get hidden activations of the first RBM to train the second.
        num_samples = data.shape[0]
   
        first_hidden_activations = self.first_rbm.run_visible(data)
            
        self.second_rbm.train(first_hidden_activations,learning_rate=learning_rate,batch_size=batch_size,k=k)
        
    # Generate data seeded by some sample
    def daydream(self,num_samples,initialising_data):
    
        samples = np.ones((num_samples, self.first_rbm.num_visible + 1))
        
        samples[0,1:] = initialising_data
        
        for i in range(1, num_samples):
          visible = samples[i-1,:]

          # Calculate the activations of the hidden units.
          first_hidden_activations = np.dot(visible, self.first_rbm.weights)
          # Calculate the probabilities of turning the hidden units on.
          first_hidden_probs = self._logistic(first_hidden_activations)
          # Turn the hidden units on with their specified probabilities.
          first_hidden_states = first_hidden_probs > np.random.rand(self.first_rbm.num_hidden + 1)
          # Always fix the bias unit to 1.
          first_hidden_states[0] = 1
          
          second_hidden_activations = np.dot(first_hidden_states, self.second_rbm.weights)
          second_hidden_probs = self._logistic(second_hidden_activations)
          second_hidden_states = second_hidden_probs > np.random.rand(self.second_rbm.num_hidden + 1)
          second_hidden_states[0] = 1

          # Recalculate the probabilities that the first hidden layer units are on.
          first_hidden_activations = np.dot(second_hidden_states, self.second_rbm.weights.T)
          first_hidden_probs = self._logistic(first_hidden_activations)
          first_hidden_states = first_hidden_probs > np.random.rand(self.second_rbm.num_visible + 1)
          
          # Recalculate the probabilities that the visible units are on.
          visible_activations = np.dot(first_hidden_states, self.first_rbm.weights.T)
          visible_probs = self._logistic(visible_activations)
          visible_states = visible_probs > np.random.rand(self.first_rbm.num_visible + 1)
          samples[i,:] = visible_states

        # Ignore the bias units (the first column), since they're always set to 1.
        return samples[:,1:]

    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))
    

# Stacking two RBMS - this time training the second rbm to reproduce the visible layer
# Instead of the hidden layer of the first rbm

class Stacked_RBM_GenVis:

    def __init__(self, num_visible, num_first_hidden, num_second_hidden):
        self.first_rbm = RBM(num_visible = num_visible, num_hidden = num_first_hidden)
        self.second_rbm = RBM(num_first_hidden,num_second_hidden)
            
    def train(self,data,epochs=100,learning_rate=0.1,batch_size=100,k=3):
    
        self.first_rbm.train(data,epochs=epochs,learning_rate=learning_rate,batch_size=batch_size,k=k)
        
        # Want to get hidden activations of the first RBM to train the second.
        num_samples = data.shape[0]
    
        first_hidden_activations = self.first_rbm.run_visible(data)
                
        self.second_rbm.train(first_hidden_activations,learning_rate=learning_rate,batch_size=batch_size,k=k)
            
    # Generate data seeded by some sample
    def daydream(self,num_samples,initialising_data):
        
        samples = np.ones((num_samples, self.first_rbm.num_visible + 1))
            
        samples[0,1:] = initialising_data
            
        for i in range(1, num_samples):
            visible = samples[i-1,:]

            # Calculate the activations of the hidden units.
            first_hidden_activations = np.dot(visible, self.first_rbm.weights)
            # Calculate the probabilities of turning the hidden units on.
            first_hidden_probs = self._logistic(first_hidden_activations)
            # Turn the hidden units on with their specified probabilities.
            first_hidden_states = first_hidden_probs > np.random.rand(self.first_rbm.num_hidden + 1)
            # Always fix the bias unit to 1.
            first_hidden_states[0] = 1
              
            second_hidden_activations = np.dot(first_hidden_states, self.second_rbm.weights)
            second_hidden_probs = self._logistic(second_hidden_activations)
            second_hidden_states = second_hidden_probs > np.random.rand(self.second_rbm.num_hidden + 1)
            second_hidden_states[0] = 1
            
            # Recalculate the probabilities that the first hidden layer units are on.
            first_hidden_activations = np.dot(second_hidden_states, self.second_rbm.weights.T)
            first_hidden_probs = self._logistic(first_hidden_activations)
            first_hidden_states = first_hidden_probs > np.random.rand(self.second_rbm.num_visible + 1)
              
            # Recalculate the probabilities that the visible units are on.
            visible_activations = np.dot(first_hidden_states, self.first_rbm.weights.T)
            visible_probs = self._logistic(visible_activations)
            visible_states = visible_probs > np.random.rand(self.first_rbm.num_visible + 1)
            samples[i,:] = visible_states

        # Ignore the bias units (the first column), since they're always set to 1.
        return samples[:,1:]

    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))
        


