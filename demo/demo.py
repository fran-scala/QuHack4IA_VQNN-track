from colorama import Fore, Back, Style
import numpy as np
import pennylane as qml
import jax
import jax.numpy as jnp
from sklearn.preprocessing import MinMaxScaler

n_qubits = 7
layers = 2
sublayers = 4
params_per_sublayer = 3 * n_qubits
device = qml.device("default.qubit.jax", wires=n_qubits)
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))


def print_banner():
    filepath = './banner.txt'
    with open(filepath) as fp:
        line = fp.readline()
        while line:
            print(Fore.GREEN + "{}".format(line), end='')
            line = fp.readline()

    print("\n\n" + Fore.CYAN + "--Welcome to Compress Bot--")
    print(Fore.CYAN + "Hi, I am Compression Bot, I can help you calculate the force required to reach the compression point from your mixed ingredients.")


def do_preprocessing(X):
    # Perform min-max scaling
    #min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_scaled = min_max_scaler.fit_transform(X.reshape(-1, 1))
    return x_scaled


def rev_min_max_func(scaled_val):
    max_val = 82.6
    min_val = 2.33
    og_val = (scaled_val * (max_val - min_val)) + min_val
    return og_val
    """x_inverse = min_max_scaler.inverse_transform(scaled_val.reshape(-1, 1))
    return x_inverse"""


def encode_data_nonlinear(x):
    '''
    x: input features
    encoding using arctan(x)

    (x must be in [-1,1])
    '''
    for i in np.arange(n_qubits):
        qml.RX(jnp.arctan(x[i]) * 2, wires=i)
        qml.RZ(x[i] ** 2 * jnp.pi, wires=i)


def basic_ansatz_layer(thetas):
    '''
    thetas - Parameters of the QNN

    Creates the basic ansatz of the QNN. This is composed of a RX, RY and RZ rotation on each qubit,
    followed by CNOTs gates on neighbouring qubits in linear chain.
    '''
    k = 0

    for i in range(n_qubits):
        qml.RX(thetas[i], wires=i)
    k += n_qubits

    for i in range(n_qubits):
        qml.RY(thetas[i + k], wires=i)
    k += n_qubits

    for i in range(n_qubits):
        qml.RZ(thetas[i + k], wires=i)
    k += n_qubits

    for i in range(0, n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])  #


def basic_ansatz(thetas, ):
    '''
    thetas - Parameters of the QNN

    Applies the Anstaz to the circuit.
    The ansatz is composed by a repition of sublayers.
    '''
    k = 0
    for lay in range(sublayers):
        basic_ansatz_layer(thetas[k:params_per_sublayer + k])
        k += params_per_sublayer


@qml.qnode(device, interface='jax')
def circuit_nonlinear(x, thetas):
    '''
    x - input features
    thetas - Parameters of thn QNN

    Creates the entire circuit. A layer is composed of data encoding layer followed by a number of ansatz layers.
    The ansatz is formed by a repetition of the sublayer.
    '''
    j = 0
    for lay in range(layers):
        encode_data_nonlinear(x, )
        qml.Barrier(wires=range(n_qubits), only_visual=True)
        basic_ansatz(thetas[j:j + (params_per_sublayer) * sublayers], )
        j += (params_per_sublayer) * sublayers
        qml.Barrier(wires=range(n_qubits), only_visual=True)
    qml.Barrier(wires=range(n_qubits), only_visual=True)

    return qml.expval(qml.PauliZ(0))


def load_model():
    dir_path = '..'
    data = dir_path + f'/results/nonlinear_with_outliers/{layers}l-{sublayers}p'
    opt_params = np.load(data + '/opt_params.npy', )

    qnn_batched = jax.vmap(circuit_nonlinear, (0, None,))
    return qnn_batched, opt_params


def get_input_data():
    print('\n' + Fore.CYAN + 'Now give me some important ingredients related to your mixture:' + Fore.WHITE)

    cement = ""
    blast_furnace_slag = ""
    water = ""
    superplasticizer = ""
    coarse_aggregate = ""
    fine_aggregate = ""
    age = ""

    while cement == "":
        print("Cement (kg/m3): ", end="")
        cement = input()
    while blast_furnace_slag == "":
        print("Blast furnace slag (kg/m3): ", end="")
        blast_furnace_slag = input()
    while water == "":
        print("Water (kg/m3): ", end="")
        water = input()
    while superplasticizer == "":
        print("Superplasticizer (kg/m3): ", end="")
        superplasticizer = input()
    while coarse_aggregate == "":
        print("Coarse aggregate (kg/m3): ", end="")
        coarse_aggregate = input()
    while fine_aggregate == "":
        print("Fine aggregate (kg/m3): ", end="")
        fine_aggregate = input()
    while age == "":
        print("Age (days): ", end="")
        age = input()

    return np.array([float(cement), float(blast_furnace_slag), float(water), float(superplasticizer),
                     float(coarse_aggregate), float(fine_aggregate), float(age)])


def main():
    print_banner()
    model, params = load_model()
    X = get_input_data()
    X_final = do_preprocessing(X)
    #print(X_final.reshape(1, -1))
    print('\n' + Fore.MAGENTA + 'Calculate your result...')
    y = model(X_final.reshape(1, -1), params)
    #print(y)
    y_final = rev_min_max_func(y)[0]
    #print('\n' + Fore.RED + 'This is your result: ' + Fore.WHITE + str(round(y_final[0], 2)) + ' MPa')
    print('\n' + Fore.RED + 'This is your result: ' + Fore.WHITE + str(round(abs(y_final), 2)) + ' MPa')


if __name__ == "__main__":
    main()
