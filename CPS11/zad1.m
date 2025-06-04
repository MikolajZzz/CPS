clear all;
close all;
clc;
a = 0.9545;  % współczynnik predykcji
[x, Fs] = audioread('DontWorryBeHappy.wav');  % wczytanie sygnału audio (domyślnie [-1,1])
x = x(:,1);  % jeżeli stereo, wybieramy pierwszy kanał
N = length(x);
y = zeros(size(x));    % wektor odtworzonego sygnału DPCM
d = zeros(size(x));    % wektor sygnału różnicy
d_q = zeros(size(x));  % wektor skwantowanych różnic

% Inicjalizacja: przyjmujemy y(0)=0 (lub y(1)=x(1) w wersji bez kwantyzacji pierwszej próbki)
dmin = min(x - a * [0; x(1:end-1)]);
dmax = max(x - a * [0; x(1:end-1)]);
for n = 1:N
    if n == 1
        y_prev = 0;
    else
        y_prev = y(n-1);
    end
    % Oblicz błąd predykcji i kwantyzuj:
    d(n) = x(n) - a * y_prev;
    % Używamy tej samej funkcji kwantyzacji (zakładamy globalny zakres z całego sygnału)
    d_q(n) = lab1_kwant(d(n),dmin,dmax);
    % Rekonstrukcja sygnału:
    y(n) = a * y_prev + d_q(n);
end

% Porównanie sygnałów:
figure;
plot((1:N)/Fs, x, 'b', (1:N)/Fs, y, 'r--');
xlabel('Czas [s]'); ylabel('Amplituda');
legend('Oryginał x(n)', 'Rekonstrukcja y(n)');
title('Porównanie sygnału oryginalnego i po DPCM (4-bit)');
grid on;

% Obliczanie błędu i jakości:
error = x - y;
mse = mean(error.^2);
fprintf('Błąd średniokwadratowy (MSE): %.6f\n', mse);

%%Dodatkowe 
bits16 = ADPCM_encoder(x, 4);   % 4-bit ADPCM
y16 = ADPCM_decoder(bits16, 4);

bits32 = ADPCM_encoder(x, 2);   % 2-bit ADPCM
y32 = ADPCM_decoder(bits32, 2);

% Porównanie jakości oryginału x i odtworzonych sygnałów y16, y32
figure;
subplot(2,1,1);
plot((1:N)/Fs, x, 'b', (1:N)/Fs, y16, 'g--');
title('16 kb/s ADPCM G.726'); legend('oryginał','dekod. 16 kb/s'); grid on;
subplot(2,1,2);
plot((1:N)/Fs, x, 'b', (1:N)/Fs, y32, 'm--');
title('32 kb/s ADPCM G.726'); legend('oryginał','dekod. 32 kb/s'); grid on;
function d_q = lab1_kwant(d,dmin1,dmax1)
    % lab1_kwant - Równomierna kwantyzacja sygnału d do 16 poziomów (4 bity)
    % Argument d może być wektorem. Funkcja zwraca wektor d_q tej samej długości.
    L = 16;                       % liczba poziomów kwantyzacji
    dmin = dmin1;           % najmniejsza wartość sygnału
    dmax = dmax1;            % największa wartość
    if dmax==dmin
        % Jeżeli sygnał jest stały, zwracamy ten sam sygnał (brak błędu)
        d_q = d;
        return;
    end
    delta = (dmax - dmin) / (L-1);    % szerokość przedziału kwantyzacji
    % Mapujemy wartość d do indeksu poziomu [0..L-1]
    % floor(x/Δ + 0.5) jest równoważne zaokrągleniu do najbliższego
    idx = floor((d - dmin) / delta + 0.5);
    % Klipping indeksów, aby pozostały w przedziale [0, L-1]
    idx(idx < 0) = 0;
    idx(idx > L-1) = L-1;
    % Odtworzone wartości na poziomach środkowych
    d_q = dmin + idx * delta;
end
function x = ADPCM_decoder(adpcm, numBits)
    % adpcm: wektor kodów ADPCM (0..2^numBits-1)
    % numBits: liczba bitów na próbkę (2 lub 4)
    N = length(adpcm);
    x = zeros(N,1);         % wyjściowe próbki
    ss = 1.0;               % początkowy krok kwantyzacji (dobieralny)
    prev = 0;               % poprzednia próbka (początkowo 0)
    for i = 1:N
        code = adpcm(i);
        if numBits == 4
            % Rozpakuj bity (4-bitowy kod: B3=znak, B2-B0=wielkość)
            B3 = floor(code/8);               
            B2 = floor(mod(code,8)/4);
            B1 = floor(mod(code,4)/2);
            B0 = mod(code,2);
            % Odtwórz różnicę
            diff = (B2*ss + B1*(ss/2) + B0*(ss/4) + ss/8);
            if B3==1, diff = -diff; end
        else
            % Rozpakuj 2-bitowy kod: załóżmy: pierwszy bit = znak, drugi = wielkość
            B1 = floor(code/2);  % znak (0 lub 1)
            B0 = mod(code,2);    % wielkość (0 lub 1)
            % Przyjmujemy poziomy ±0.5 i ±1.5 kroku (mid-rise z 4 poziomami)
            mag = 0.5 + B0*1.0;    % 0->0.5, 1->1.5
            diff = mag * ss;
            if B1==1, diff = -diff; end
        end
        % Predykcja i sumowanie:
        curr = prev + diff;
        x(i) = curr;
        prev = curr;
        % Adaptacja kroku kwantyzacji (przykładowa reguła):
        % Dla 4-bit: jak w tabeli ADPCM. Dla 2-bit: prosta reguła.
        if numBits == 4
            % Określamy M na podstawie bitów wielkości B2,B1,B0
            magBits = B2*4 + B1*2 + B0;
            if magBits >= 4
                M = 2*(magBits-3);   % np. 4->2,5->4,6->6,7->8
            else
                M = -1;              % dla 0..3 M = -1
            end
            ss = ss * (1.1^M);
        else
            % Dla 2-bit: gdy B0=1 (duża zmiana) zwiększ krok, inaczej zmniejsz
            if B0 == 1
                ss = ss * 1.2;
            else
                ss = ss * 0.9;
            end
        end
        % ogranicz zakres kroku do pewnych granic
        ss = max(min(ss,100), 0.01);
    end
end
function adpcm = ADPCM_encoder(x, numBits)
    N = length(x);
    adpcm = zeros(N,1);  % wektor zakodowany
    ss = 1.0;            % początkowy krok kwantyzacji
    prev = 0;            % poprzednia próbka (dla predykcji)
    
    for i = 1:N
        diff = x(i) - prev;
        if numBits == 4
            % Kwantyzacja 4-bitowa
            signBit = diff < 0;
            diff = abs(diff);
            % Oblicz bity wielkości
            mag = floor(diff / ss);  % aproksymacja
            if mag > 7, mag = 7; end
            % Rozkład na 3 bity
            B2 = bitget(mag,3);
            B1 = bitget(mag,2);
            B0 = bitget(mag,1);
            code = signBit*8 + B2*4 + B1*2 + B0;
        else
            % Kwantyzacja 2-bitowa
            signBit = diff < 0;
            diff = abs(diff);
            mag = (diff >= ss); % 0: mała zmiana, 1: duża
            code = signBit*2 + mag;
        end
        adpcm(i) = code;
        
        % Dekodowanie różnicy (symetrycznie z dekoderem)
        if numBits == 4
            B3 = floor(code/8);               
            B2 = floor(mod(code,8)/4);
            B1 = floor(mod(code,4)/2);
            B0 = mod(code,2);
            d_hat = (B2*ss + B1*(ss/2) + B0*(ss/4) + ss/8);
            if B3==1, d_hat = -d_hat; end
        else
            B1 = floor(code/2);
            B0 = mod(code,2);
            mag = 0.5 + B0*1.0;
            d_hat = mag * ss;
            if B1==1, d_hat = -d_hat; end
        end
        
        % Rekonstrukcja dla predykcji
        curr = prev + d_hat;
        prev = curr;
        
        % Adaptacja kroku kwantyzacji
        if numBits == 4
            magBits = B2*4 + B1*2 + B0;
            if magBits >= 4
                M = 2*(magBits-3);
            else
                M = -1;
            end
            ss = ss * (1.1^M);
        else
            if B0 == 1
                ss = ss * 1.2;
            else
                ss = ss * 0.9;
            end
        end
        ss = max(min(ss,100), 0.01);
    end
end
