import numpy as np
import matplotlib.pyplot as plt

# Clase neurona
# * Métodos: __init__, forward(entrada neurona) -> calculo neurona, backward(Gradiente de la salida) -> Gradiente de la entrada, actualizarPesos(learning_rate)
class Neurona():
    def __init__(self):
        self.pesos = np.random.uniform(-1, 1)
        self.bias = 0
        self.d_weights = 0  # Gradiente de los pesos
        self.d_bias = 0     # Gradiente del sesgo
        self.input = 0      # Input

    def forward(self, x):
        self.input = x
        salida = x * self.pesos + self.bias
        return leakyReLU(salida)
    
    # ! dL_da: Gradiente de la salida / 
    def backward(self, dL_da):

        # ! da_dz: Derivada de la función de activación (Leaky ReLU)
        da_dz = 1 if self.input > 0 else 0.01

        # ! dL_dz: Gradiente de la pérdida respecto a la salida
        dL_dz = dL_da * da_dz

        self.d_weights = dL_dz * self.input
        self.d_bias = dL_dz

        # ! dL_din: Gradiente de la pérdida respecto a la entrada (para backpropagation)
        dL_din = dL_dz * self.pesos
        return dL_din
    
    def actualizarPesos(self, learning_rate):
        self.pesos -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias

# Clase capa salida
# * Métodos: __init__(numero de neuronas ocultas), forward(array salidas anterior capa) -> resultado en farenheit, backward(Gradiente de la salida) -> Gradiente de la entrada, actualizarPesos(learning_rate)
class CapaSalida():
    def __init__(self, num_neuronas):
        self.pesos = np.random.uniform(-1, 1, num_neuronas)
        self.bias = 0
        self.d_weights = np.zeros(num_neuronas)         # Gradiente de los pesos
        self.d_bias = 0                                 # Gradiente del sesgo
        self.inputs = []                                # Guardar las entradas para el backward
    
    def forward(self, salidas):
        salidas = np.array(salidas)
        self.inputs = salidas
        entradas = np.array([])
        for salida in salidas:
            entradas = np.append(entradas, salida)
        
        # ! Operador @ es el producto vectorial
        out = entradas @ self.pesos + self.bias

        return out

    def backward(self, dL_dypred):

        self.d_weights = dL_dypred * self.inputs
        self.d_bias = dL_dypred

        # ! dL_din: Gradiente de la pérdida respecto a la entrada (para backpropagation)
        dL_din = dL_dypred * self.pesos

        return dL_din
    
    def actualizarPesos(self, learning_rate):
        self.pesos -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias

# Obtener datos de fichero csv y devuelve dos arrays cels y fars
# ? Input: String de archivo csv
# * Output: Dos arrays
def getDatos(archivo):
    datos = np.loadtxt(archivo, delimiter=',')
    cel = datos[:,0]
    far = datos[:,1]
    return cel, far

# Normalizar los datos de los arrays dados
# ? Input: Dos arrays
# * Output: Dos arrays normalizados
def normalizarDatos(cels, fars):
    mediaC = np.mean(cels)
    stdC = np.std(cels)
    mediaF = np.mean(fars)
    stdF = np.std(fars)

    celN = []
    farN = []

    for i in range(len(cels)):
        celN.append((cels[i] - mediaC) / stdC) 
        farN.append((fars[i] - mediaF) / stdF)

    print('Datos normalizados!')    
    return celN , farN

# Separar los datos en train y test dado un porcentaje que corta los datos
# ? Input: Dos arrays normalizados y un float (~0.75)
# * Output: Dos arrays de tamaños len(array) * porcentaje y len(array) * (1 - porcentaje)
def trainTestDataSplit(cels, faren, porcentaje):
    split = int(len(cels) * porcentaje)

    trainData = []
    testData = []

    contador = 0
    while contador <= len(cels) - 1:
        temp = [cels[contador], faren[contador]]
        if contador < split:
            trainData.append(temp)
        else:
            testData.append(temp)
        contador += 1

    print('Datos separados en train y test!')
    return trainData, testData

# Inicializar las neuronas con peso y bias aleatorios
# ? Input: Int de neuronas
# * Output: Array de neuronas de len = Input y la capa de salida
def iniciarNeuronas(num_neuronas):
    neuronas = []
    for i in range(num_neuronas):
        neuronas.append(Neurona())

    capaSalida = CapaSalida(num_neuronas)

    print('Neuronas inicializadas!')
    return neuronas, capaSalida

# Aplicar funcion de activacion: Leaky ReLU (Rectified Linear Unit) 
# ? Input: Numero a aplicar la funcion
# * Output: x si x > 0, 0.01x si x < 0
def leakyReLU(x):
    if x > 0:
        return x
    else:
        return 0.01 * x

# Cálculo de la pérdida usando MSE (Mean Squared Error), L = 1/2 * (y_pred - y_real)^2
# ? Input: Dato real y el predicho
# * Output: Error y pérdida
def MSE(y_real, y_pred):
    error = y_pred - y_real
    loss = 0.5 * error**2

    return error, loss

# Propagar el error de la ultima capa a la oculta para el ajuste de pesos y biases
# ? Input: float Celsius, float Farenheit,array neuronas, array neuronas.fw, objeto capa salida, learning rate (0.1)
# * Output: Pérdida
def backpropagation(y_real, neuronas, neuronasFw, capa_salida, learning_rate):

    prediccion = capa_salida.forward(neuronasFw)

    error, perdida = MSE(y_real, prediccion)

    # Perdida en la capa de salida
    dL_din_salida = capa_salida.backward(error) 

    # Backpropagation en las neuronas ocultas
    dL_din_lista = []  
    for i in range(len(neuronasFw)):
        dL_din = neuronas[i].backward(dL_din_salida[i])
        dL_din_lista.append(dL_din)  # Almacenar el gradiente

    # Actualizar pesos y biases
    for neurona in neuronas:
        neurona.actualizarPesos(learning_rate)
        capa_salida.actualizarPesos(learning_rate)

    return perdida

# Entrena la red neuronal
# ? Input: Array de trainData, array de neuronas, objeto capa salida, learning rate (0.1), epochs (1000)
# * Output: Array de pérdidas
def fit(neuronas, capa_salida, trainData, learning_rate, epochs):
    for i in range(epochs):
        for j in range(len(trainData) - 1):
            gradosCelsi = trainData[j][0]
            gradosFaren = trainData[j][1]

            neuroFw = []
            for neurona in neuronas:
                neuro = neurona.forward(gradosCelsi)
                neuroFw.append(neuro)

            perdida = backpropagation(gradosFaren, neuronas, neuroFw, capa_salida, learning_rate)

        if (i % 100) == 0:
            print(f'Epoch: {i}, Pérdida: {perdida}')
    
    print('Entrenamiento finalizado!')


# Predice el valor en farenheit de un valor en celsius
# ? Input: Array de neuronas, objeto capa salida, float celsius, array celsRaw, array farRaw
# * Output: String con el valor en farenheit
def predict(neuronas, capa_salida, celsius, celsRaw, farRaw, predictManual = True):
    if predictManual:
        print(f'{celsius}ºC son: ', end='')
    celsRaw = np.array(celsRaw)
    
    mediaC = np.mean(celsRaw)
    stdC = np.std(celsRaw)
    celsius = (celsius - mediaC) / stdC
    neuronasFw = []
    for neurona in neuronas:
        neuro = neurona.forward(celsius)
        neuronasFw.append(neuro)
    
    farenheitN = capa_salida.forward(neuronasFw)

    mediaF = np.mean(farRaw)
    stdF = np.std(farRaw)
    farenheit = round(farenheitN * stdF + mediaF, 1)

    return farenheit

def verDatos(neuronas, capa_salida):
    print('Pesos neuronas y bias:')
    contador = 1
    for neurona in neuronas:
        print("w" + str(contador) + f":{neurona.pesos}")
        print("b" + str(contador) + f":{neurona.pesos}")
        print()
        contador += 1

    print('Pesos capa salida:')
    print(f"w_out: {capa_salida.pesos}")
    print(f"b_out: {capa_salida.bias}")

def testearRed(neuronas, capa_salida, testData, celsRaw, farRaw):
    accuracy = 0
    aciertos = 0
    for i in range(len(testData) - 1):
        celsius = testData[i][0]
        farenheit = testData[i][1]
        
        mediaC = np.mean(celsRaw)
        stdC = np.std(celsRaw)
        celsius = round(celsius * stdC + mediaC, 1)

        farenheitPred = predict(neuronas, capa_salida, celsius, celsRaw, farRaw, False)

        mediaF = np.mean(farRaw)
        stdF = np.std(farRaw)
        farenheit = round(farenheit * stdF + mediaF, 1)

        print(f'{celsius}ºC son: {farenheitPred}ºF, Real: {farenheit}ºF')
        if farenheitPred == round(farenheit, 1):
            aciertos += 1
        
        accuracy = aciertos / (i + 1)

        print(f'Accuracy: {accuracy}')
        
        

    
    