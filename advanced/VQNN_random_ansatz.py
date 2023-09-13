import numpy as np
import pennylane as qml
import jax
import optax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches
import pandas as pd
import os

# constant
n_qubits = 7  # equal to the number of features
layers = 3
sublayers = 1
params_per_sublayer = 3 * n_qubits
seed = 1234
device = qml.device("default.qubit.jax", wires=n_qubits)


def encode_data_nonlinear(x):
    '''
    x: input features
    encoding using arctan(x)

    (x must be in [-1,1])
    '''
    for i in np.arange(n_qubits):

        qml.RX(jnp.arctan(x[i])*2, wires=i)
        qml.RZ(x[i]**2*jnp.pi, wires=i)


@jax.jit
def calculate_mse_cost(X, y, theta, qnn):
    '''
    X - Dataset
    y - The true label associated with X
    theta - Parameters of the QNN

    Calculates the MSE cost for a given data point and label.
    '''
    y = jnp.array(y)
    yp = qnn(X, theta)
    cost = jnp.mean((yp - y) ** 2)  #
    return cost


# Optimization update step
@jax.jit
def optimizer_update(opt_state, params, x, y, optimizer, qnn):
    '''
    opt_state - state of the optimizer
    params - Parameters of the QNN
    x - input features
    y - The true label associated with X

    Calculates the MSE cost and update the parameters according to the chosen optimizer.
    '''
    loss, grads = jax.value_and_grad(lambda theta: calculate_mse_cost(x, y, theta, qnn))(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@qml.qnode(device,interface='jax')
def circuit_random(x, thetas):
    '''
    x - input features
    thetas - Parameters of thn QNN

    Creates the entire circuit. A layer is composed of data encoding layer followed by a number of ansatz layers.
    The ansatz is formed by a repetition of the sublayer.
    '''
    j = 0
    for lay in range(layers):
        encode_data_nonlinear(x,)
        qml.Barrier(wires=range(n_qubits),only_visual=True)
        
        qml.RandomLayers(thetas[j], wires=range(n_qubits), seed=seed+j)
        j += 1
        qml.Barrier(wires=range(n_qubits),only_visual=True)
    qml.Barrier(wires=range(n_qubits),only_visual=True)

    return qml.expval(qml.PauliZ(0))

# import matplotlib.pyplot as plt
# layers = 3
# sublayers = 1
# params_per_sublayer = n_qubits
# key = jax.random.PRNGKey(seed)
# initial_params = jax.random.normal(key, shape=(layers ,sublayers,params_per_sublayer))
# print(initial_params)
# qml.draw_mpl(circuit_random,expansion_strategy='device')([1,0,1,1,1,1,0,0],initial_params,)
# plt.savefig('./random_circuit.pdf')

@jax.jit
def calculate_mse_cost(X, y, theta):
    """
    X - Dataset
    y - The true label associated with X
    theta - Parameters of the QNN

    Calculates the MSE cost for a given data point and label.
    """
    y = jnp.array(y)
    yp = qnn(X, theta)
    cost = jnp.mean((yp - y) ** 2)  #
    return cost


# Optimization update step
@jax.jit
def optimizer_update(opt_state, params, x, y, ):
    """
    opt_state - state of the optimizer
    params - Parameters of the QNN
    x - input fetures
    y - The true label associated with X

    Calculates the MSE cost and updatez the parameters according to the chosen optimizer.
    """
    loss, grads = jax.value_and_grad(lambda theta: calculate_mse_cost(x, y, theta, ))(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# /g100/home/usertrain/a08tra20
df_with_outliers = pd.read_csv('./QuHack4IA_VQNN-track/dataset/dataset_with_outliers_without_feature.csv')

X = df_with_outliers.drop(columns=["concrete_compressive_strength"]).values
y = df_with_outliers["concrete_compressive_strength"].values
seed = 1234
np.random.seed(seed)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, shuffle=True, random_state=seed, test_size=0.3)
X_test, X_valid, y_test, y_valid = train_test_split(X_valid, y_valid, shuffle=True, random_state=seed, test_size=1 / 3)

X_train = jnp.array(X_train)
X_valid = jnp.array(X_valid)
X_test = jnp.array(X_test)

print(X_train.shape, X_valid.shape, X_test.shape)

# Parameter initialization
# n_runs = 1
epochs = 60
batch_size = 30
seed = 1234
min_layers, max_layers = 1, 4
min_sublayers, max_sublayers = 1, 4

for layers in range(min_layers, max_layers + 1):
    for sublayers in range(min_sublayers, max_sublayers + 1):

        # creating a folder to save data
        dir_path = '.'
        data = dir_path+f'/results/random/{layers}l-{sublayers}p'
        os.makedirs(data, 0o755, exist_ok=True)
        # Jax jit and vmap speed up the computational times of the circuit

        params_per_sublayer = n_qubits
        qnn_batched = jax.vmap(circuit_random, (0, None,))
        qnn = jax.jit(qnn_batched)

        # Lists to save data
        costs = []
        val_costs = []
        train_per_epoch = []
        val_per_epoch = []

        # Creating the initial random parameters for the QNN
        key = jax.random.PRNGKey(seed)
        initial_params = jax.random.normal(key, shape=(layers * sublayers,1,params_per_sublayer))
        key = jax.random.split(key)[0]
        params = jnp.copy(initial_params)

        # Optimizer initialization
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(initial_params)

        for epoch in range(1,epochs+1):
            # Generation of random indices to be used for batch
            idxs_dataset = jax.random.choice(key, jnp.array(list(range(X_train.shape[0]))), shape=(X_train.shape[0],),
                                             replace=False)
            key = jax.random.split(key)[0]
            cost = 1
            val_cost = 1
            for i in gen_batches(X_train.shape[0], batch_size):
                idxs = idxs_dataset[i]

                # Calculate cost function and update parameters accordingly
                params, opt_state, _ = optimizer_update(opt_state, params, X_train[idxs, :], y_train[idxs])

                # Save MSE Costs and accuracies for both train and validation dataset
                cost = calculate_mse_cost(X_train, y_train, params)  # calculate_mse_cost_and_accuracy
                costs.append(cost)

                val_cost = calculate_mse_cost(X_valid, y_valid, params, )  # calculate_mse_cost_and_accuracy
                val_costs.append(val_cost)
            train_per_epoch.append(cost)
            val_per_epoch.append(val_cost)

            print(f"layers:{layers}, p:{sublayers}, epoch {epoch}/{epochs}", '--- Train cost:', cost, '--- Val cost:',
                  val_cost, end='\r')

        np.save(data + '/train_cost.npy', list(costs))
        np.save(data + '/val_cost.npy', list(val_costs))
        np.save(data + '/train_cost_per_epoch.npy', list(train_per_epoch))
        np.save(data + '/val_cost_per_epoch.npy', list(val_per_epoch))
        np.save(data + '/opt_params.npy', list(params))




##########
# PREDICTIONS
############


params_per_sublayer = 3*n_qubits 

min_layers, max_layers = 1,4
min_sublayers, max_sublayers = 1,4
d_mse = {}
print('RANDOM ANSATZ')
for layers in range(min_layers, max_layers+1):
    for sublayers in range(min_sublayers, max_sublayers+1):
        # re-create the circuit
        qnn_batched = jax.vmap(circuit_random, (0, None,))
        qnn = jax.jit(qnn_batched)

        # load optimal parameters
        dir_path = './QuHack4IA_VQNN-track'
        data = dir_path+f'/results/nonlinear_random_ans_with_outliers/{layers}l-{sublayers}p' 
        opt_params = np.load(data+'/opt_params.npy',)

        # predict on validation
        y = jnp.array(y_valid)
        yp = qnn(X_valid, opt_params) 
        mse_cost = jnp.mean((yp - y) ** 2)
        
        print(layers, sublayers, mse_cost)
        d_mse[(layers, sublayers)] = [mse_cost]

# pd.DataFrame(d_mse).to_csv(dir_path+f'/results/basic_entangler/val_mse_no_outliers.csv', )

############
# TESTING ON THE BEST
# L2S4 is the best with_outliers dataset
############

best_l, best_s = 0, 0
best_v = 1
for k,v in d_mse.items():
    if v[0] < best_v:
        best_v = v[0]
        best_l, best_s = k

print(f'BEST VAL:{best_l}L, {best_s}S', best_v)


params_per_sublayer = 3*n_qubits 

layers = best_l
sublayers = best_s

# re-create the circuit
qnn_batched = jax.vmap(circuit_random, (0, None,))
qnn = jax.jit(qnn_batched)

# load optimal parameters
dir_path = './QuHack4IA_VQNN-track/'
data = dir_path+f'/results/nonlinear_random_ans_with_outliers/{layers}l-{sublayers}p' 
opt_params = np.load(data+'/opt_params.npy',)

# predict on validation
y = jnp.array(y_test)
yp = qnn(X_test, opt_params) 

mse_cost = jnp.mean((yp - y) ** 2)

print(f'BEST TEST:{best_l}L, {best_s}S', mse_cost)