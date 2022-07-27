import scipy as sc  #Paq. p-análisis numérico 
import numpy as np  #Paq. p-cálculo numérico

def Suma (x): #La función recibira una variable de tipo array
    suma = 0 #Definimos el valor inicial de la suma en 0
    for i in range(len(x)): #iniciamos un ciclo for durante el cual se realizará la operación
        suma += abs(x[i])**2 #iterativamente sumamos el valor anterior de la variable más el elemento i del array
    return suma
def p_interior(f,g): #Definimos nuestra función para dos arrays f y g que deberán tener la misma longitud de puntos
    resultado = 0 #Nombramos a la variable donde almacenaremos el resultado con su valor inicial
    for i in range(len(f)): #Establecemos un bucle para las operaciones iterativas
        resultado += f[i]*g[i] #Obtenemos el producto de los elementos f_i y g_i y lo sumamos iterativamente
    return resultado #La función arroja el resultado del producto interior
def conv_tiempo(f,g): #Definimos nuestra función para dos arrays f y g  
    n,m = len(f),len(g) #Obtenemos las longitudes de las series
    k = int(m/2) #Obtenemos el número de elementos a añadir en f para el encamado
    '''Encamado de la serie f'''
    f_pad = np.pad(f,(k,k),'constant') #Encamamos la serie usando el módulo de numpy pad()
    #El primer parámetro es el array a encamar, el segundo es la longitud de puntos a añadir al inicio 
    #y al final (k,k), el último parámetro indica que los valores serán constantes igual a 0
    '''Proceso de convolución'''
    p=n+m-1 #Longitud del resultado de la convolución
    conv = np.zeros(p) #Definimos el array donde se almacenarán los resultados de los productos punto
    for ti in range(len(f_pad)-m): #Establecemos el ciclo iterativo para los productos interiores
        f_temp = f_pad[ti:ti+m] #Extraemos el segmento de la serie f correspondiente
        g_temp = np.flip(g) #Invertimos la serie g 
        conv[ti+k] = p_interior(f_temp,g_temp) #Realizamos el producto interior y lo almacenamos
    '''Recortado del resultado'''
    conv_cut = conv[k:-k+1]
    return conv_cut
def cuadrado(m): #Función para crear una serie de pulso cuadrado con longitud m 
    pulso = np.ones(m) #Definimos que el pulso tendrá m número de unos
    pulso = np.pad(pulso,(10,10),'constant') #Establecemos que el pulso tendrá 10 valores 0 a los costados
    return pulso
def conv_v2(f,g): #Definimos la función para dos arrays f y g
    'Realizamos un padding a las series para que tengan el mismo número de puntos que el del resultado de la conv. n+m-1'
    n,m =len(f),len(g)
    f_pad = np.pad(f,(0,m-1),'constant')
    g_pad = np.pad(g,(0,n-1),'constant')
    #En ambos casos añadimos puntos para que al final la longitud de ambas series sea igual a p=n+m-1
    '''Cambio al dominio de la frecuencia por FFT1'''
    f_fft = np.fft.fft(f_pad)
    g_fft = np.fft.fft(g_pad)
    g_fft = g_fft/g_fft.max() #Normalizamos el espectro de g para conservar las unidades de f en la convolución
    '''Teorema de la convolución'''
    conv = np.multiply(f_fft,g_fft)
    conv_res = np.fft.ifft(conv)
    '''Recorte del resultado para que tenga longitud n'''
    k = int(len(g)/2)
    conv_res = abs(conv_res[k:-k+1])
    return conv_res
def morlet(fw,n_c,twav,freqm=500): 
    #Componente oscilatoria 'Seno complejo'
    com_sin = np.exp(1j*2*np.pi*fw*twav)
    #Ventana de Gauss
    sigma = n_c/(2*np.pi*fw)
    gauss_c = np.exp(-twav**2/(2*sigma**2))
    #Normalization
    A_sigma = 1/(np.sqrt(sigma*np.sqrt(np.pi)))
    #Morlet wavelet
    wavelet = A_sigma*com_sin*gauss_c
    return wavelet
def gauss(twav,n_c,fw):
    #Ventana de Gauss
    sigma = n_c/(2*np.pi*fw)
    gauss_c = np.exp(-twav**2/(2*sigma**2))
    #Normalization
    A_sigma = 1/(np.sqrt(sigma*np.sqrt(np.pi)))
    return gauss_c*A_sigma

def MRA(sig,fi,ff,freq_int,ni,nf,wavelet=morlet,freqm=200):
    '''Generamos los vectores de los parámetros'''
    delta = ff-fi #intervalo de frecuencias de descomposición
    k = int(delta*(1/freq_int)) # número de elementos a generar
    v_frec = np.linspace(fi,ff,k) #vector de frecuencias
    v_cycles =np.logspace(np.log10(ni), np.log10(nf),k) #vector de número de ciclos
    '''Definición de variables y la matriz donde se alojarán los resultados de las convoluciones iterativas'''
    twav = np.linspace(-1,1,freqm*2) #Definimos el vector de tiempo de la wavelet
    wav = wavelet(v_frec[0],v_cycles[0],twav,freqm) 
    n,m = len(sig),len(wav) #Obtenemos las longitudes de las señales
    p = n+m-1 #El número de puntos que contiene el resultado de la convolución
    half_wav = int(np.floor(m/2)) #Número de puntos que se añaden al inicio y al final del resultado de la convolución
    s_pad = np.pad(sig,(0,m-1),'constant') #Realizamos el padding correspondiente a la señal a analizar
    s_fft = np.fft.fft(s_pad) #Calculamos la TF de la señal tras el padding
    conv_mat = np.zeros([k,n]) #Definimos el espacio matricial donde se almacenarán los resultados
    '''Operación de convolución aprovechando el teorema homónimo'''
    for i in range(k): 
        w = wavelet(v_frec[i],v_cycles[i],twav,freqm) #Wavelet correspondiente a los parámetros i de cada vector 
        w_pad = np.pad(w, (0,n-1),'constant') #Padding de la wavelet
        w_fft = np.fft.fft(w_pad)  #TF de la wavelet
        w_fft = w_fft / w_fft.max() #Normalización 
        conv_res_prim = np.multiply(s_fft,w_fft) #Multiplicación punto-a-punto
        conv_res = np.fft.ifft(conv_res_prim) #Teorema de la Convolución  
        conv_res = abs(conv_res[half_wav:-half_wav+1])**2 #Recorte del padding y obtención de la potencia
        for j in range(n):
            conv_mat[i][j] = conv_res[j] #Almacenamos el resultado en cada fila de la matriz
    return conv_mat