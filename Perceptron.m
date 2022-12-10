clear all
close all
clc

Training = xlsread('D:/Maestria_MCIC/03. Inteligencia Computacional/01. Taller 01/Data_Set.xlsx','Sheet1','A1:E48');
Tprueba = xlsread('D:/Maestria_MCIC/03. Inteligencia Computacional/01. Taller 01/Data_Set.xlsx','Sheet1','A49:E80');


Y=Training(:,5);

X1=Training(:,1);
X2=Training(:,2);
X3=Training(:,3);
X4=Training(:,4);
alpha=1;
b=1;

W1=rand;
W2=rand;
W3=rand;
W4=rand;

Y_2=Training(:,5);

Xa=Training(:,1);
Xb=Training(:,2);
Xc=Training(:,3);
Xd=Training(:,4);
alpha_2=1;
b_2=1;



%% Llamado de la funcion para iteración

[iter,a,wa,wb,wc,wd,we]= percep(X1,X2,X3,X4,Y,W1,W2,W3,W4,b,alpha);


%% Prueba de la funcion con pesos calculados

[iter2,a2,wa2,wb2,wc2,wd2,we2]= segundoPercep(X1,X2,X3,X4,Y,W1,W2,W3,W4,b,alpha);

%% Primer entrenamiento de la función con alpha

function [iter,a,wa,wb,wc,wd,we]= percep(X1,X2,X3,X4,Y,W1,W2,W3,W4,b,alpha)


w0i = b;
w1i = W1;
w2i = W2;
w3i = W3;
w4i = W4;

%% Variables auxiliares

iteraciones = 0;

a = [0 0 0 0];


for p=1:1:100000

%while ~isequal(Y,a)

iteraciones = iteraciones + 1;

for i = 1:1:length(Y)

u = (X1(i)*w1i) + (X2(i)*w2i) + (X3(i)*w3i) + (X4(i)*w4i) + w0i;

%% Funcion de activacion

if u >= 3
    a(i) = max(Y);
else
    a(i) = 0;
end

%% Calculo de error
if Y(i) ~= a(i)

    e = alpha*(Y(i) - a(i));
    
    
    w1i= w1i + (e*X1(i));
    w2i= w2i + (e*X2(i));
    w3i= w3i + (e*X3(i));
    w4i= w4i + (e*X4(i));
    
    w0i= w0i + e;

end 
end 

end

iter=iteraciones;
wa=w0i;
wb=w1i;
wc=w2i;
wd=w3i;
we=w4i;

end



%% Seguno entrenamiento de la función con alpha

function [iter,a,wa,wb,wc,wd,we]= segundoPercep(X1,X2,X3,X4,Y,W1,W2,W3,W4,b,alpha)


w0i = b;
w1i = W1;
w2i = W2;
w3i = W3;
w4i = W4;

%% Variables auxiliares

iteraciones = 0;

a = [0 0 0 0];


for p=1:1:100000

%while ~isequal(Y,a)

iteraciones = iteraciones + 1;

for i = 1:1:length(Y)

u = (X1(i)*w1i) + (X2(i)*w2i) + (X3(i)*w3i) + (X4(i)*w4i) + w0i;

%% Funcion de activacion

if u >= 2
    a(i) = 1;
else
    a(i) = 0;
end

%% Calculo de error
if Y(i) ~= a(i)

    e = alpha*(Y(i) - a(i));
    
    
    w1i= w1i + (e*X1(i));
    w2i= w2i + (e*X2(i));
    w3i= w3i + (e*X3(i));
    w4i= w4i + (e*X4(i));
    
    w0i= w0i + e;

end 
end 

end

iter=iteraciones;
wa=w0i;
wb=w1i;
wc=w2i;
wd=w3i;
we=w4i;

end
