clear all;
close all;
clc;
% =============================
% CZĘŚĆ PODSTAWOWA (MDCT, N=32 i 128)
% =============================

% Wczytanie sygnału audio (mono) z pliku WAV
[x, fs] = audioread('DontWorryBeHappy.wav');
if size(x,2) > 1
    x = mean(x, 2);  % konwersja do mono przez uśrednienie kanałów (jeśli stereo)
end

% === MDCT+IMDCT dla N = 32 ===
N1 = 32;
X32 = mdct_analysis(x, N1);
x_rec32 = imdct_synthesis(X32, N1);
x_rec32 = x_rec32(1:length(x));     % PRZYCIĘCIE do długości oryginału

error32 = x - x_rec32;
mse32 = mean(error32.^2);
maxerr32 = max(abs(error32));
fprintf('N=%d: MSE = %.3g, max error = %.3g\n', N1, mse32, maxerr32);

% === MDCT+IMDCT dla N = 128 ===
N2 = 128;
X128 = mdct_analysis(x, N2);
x_rec128 = imdct_synthesis(X128, N2);
x_rec128 = x_rec128(1:length(x));   % PRZYCIĘCIE

error128 = x - x_rec128;
mse128 = mean(error128.^2);
maxerr128 = max(abs(error128));
fprintf('N=%d: MSE = %.3g, max error = %.3g\n', N2, mse128, maxerr128);

% === Obliczenie Q dla ~64 kb/s ===
target_bitrate = 64000;         % docelowa przepływność [bit/s]
bits_per_sample = target_bitrate / fs;   % bity na próbkę
Q = 2^bits_per_sample;          % liczba poziomów kwantyzacji (może nie być całkowita)
fprintf('I Szacowane Q (liczba poziomów) dla ~64 kb/s = %.3f (najbliżej %d poziomy)\n', Q, round(Q));





% =============================
% CZĘŚĆ DODATKOWA NR 1 – DYNAMICZNA ALOKACJA BITÓW
% =============================

N = 128;
% 1. przebieg: analiza MDCT i wyznaczenie energii współczynników w każdym bloku
X_blocks = mdct_analysis(x, N);
[num_coeffs, num_blocks] = size(X_blocks);
energy = X_blocks .^ 2;  % energia każdego współczynnika w blokach

% Obliczenie przydziału bitów dla każdego współczynnika w każdym bloku
bits_total = round(64000 / fs * N);   % docelowe bity na blok (~186 dla 128 przy 64 kb/s)
bits_alloc = zeros(num_coeffs, num_blocks);
for b = 1:num_blocks
    E = energy(:, b);
    if sum(E) == 0
        bits_alloc(:, b) = 0;  % pusty blok (cisza) – wszystkie współczynniki zero
    else
        % przydział proporcjonalny do energii
        B = bits_total * (E / sum(E));
        bits_alloc(:, b) = floor(B);  % przydział podstawowy (całkowite bity)
        % Rozdysponuj ewentualne brakujące bity (jeśli sum(floor) < bits_total) 
        diff = bits_total - sum(bits_alloc(:, b));
        if diff > 0
            % dodaj brakujące bity współczynnikom o największej energii
            [~, idx] = sort(E, 'descend');
            bits_alloc(idx(1:diff), b) = bits_alloc(idx(1:diff), b) + 1;
        end
    end
end

% 2. przebieg: kwantyzacja współczynników według wyznaczonej alokacji bitów
Xq_blocks = zeros(size(X_blocks));  % skwantyzowane współczynniki
for b = 1:num_blocks
    for k = 1:num_coeffs
        Bk = bits_alloc(k, b);
        Xval = X_blocks(k, b);
        if Bk <= 0
            Xq_blocks(k, b) = 0;  % brak bitów -> współczynnik zerowy
        else
            Q_levels = 2^Bk;
            % Zakres kwantyzacji – tu przyjmujemy ±max (tu max = |X|)
            xmax = abs(Xval);
            if xmax < 1e-12
                Xq_blocks(k, b) = 0;
            else
                % Krok kwantyzacji
                delta = (2 * xmax) / (Q_levels - 1);
                % Kwantyzacja uniform (zaokrąglenie do najbliższego poziomu)
                Xq_blocks(k, b) = sign(Xval) * min(xmax, delta * round(abs(Xval) / delta));
            end
        end
    end
end

% Synteza z użyciem skwantyzowanych współczynników
x_rec_quant = imdct_synthesis(Xq_blocks, N);

% Obliczenie błędu i SNR
x_rec_quant=x_rec_quant(1:length(x));
err = x - x_rec_quant;
mse = mean(err.^2);
signal_power = mean(x.^2);
SNR = 10 * log10(signal_power / mse);
fprintf('II Średni MSE po kwantyzacji = %.3g, SNR = %.2f dB (N=%d, 64 kb/s)\n', mse, SNR, N);

% =============================
% CZĘŚĆ DODATKOWA NR 2 – ZMIANA DŁUGOŚCI OKNA (128 <-> 32)
% =============================

x = x(:);  % wektor kolumnowy
Nx = length(x);
fs = fs;  % częstotliwość próbkowania wcześniej wczytana

% Definicja okien:
N_long = 128; N_short = 32;
L_long = 2*N_long; L_short = 2*N_short;
% Okno długie (sinusoidalne)
n_long = (0:L_long-1)';
w_long = sin((n_long + 0.5) * pi / (2*L_long));
% Okno krótkie (sinusoidalne)
n_short = (0:L_short-1)';
w_short = sin((n_short + 0.5) * pi / (2*L_short));
% Okno start (dla długości 256)
w_start = zeros(L_long, 1);
for n = 0:L_long-1
    if n < 160
        w_start(n+1) = sin((n + 0.5) * pi / (2*L_long));  % do 160 jak okno długie
    elseif n < 224
        % opadająca połowa cos od n=160 do 223
        i = n - 160;
        w_start(n+1) = cos((i + 0.5) * pi / 128);
    else
        w_start(n+1) = 0;
    end
end
% Okno stop (dla długości 256)
w_stop = zeros(L_long, 1);
for n = 0:L_long-1
    if n < 32
        w_stop(n+1) = 0;
    elseif n < 96
        % narastająca połowa sin od n=32 do 95
        i = n - 32;
        w_stop(n+1) = sin((i + 0.5) * pi / 128);
    else
        w_stop(n+1) = sin((n + 0.5) * pi / (2*L_long));
    end
end



% Symulacja kodowania z przełączaniem okien:
% Przygotuj tablicę na wyjściowe współczynniki i identyfikator typu okna
X_all = {};  % lista bloków współczynników (o zmiennej długości)
win_types = {};  % typy okien dla informacji
pos = 0;
prev_mode = 'long';  % zakładamy start z oknami długimi
output_count = 0;    % licznik wygenerowanych próbek wyjściowych
% Ustal punkty (w [próbkach] oryginalnego sygnału), gdzie nastąpi przełączenie na krótkie okna
trigger_points = [round(0.5*fs), round(1.0*fs)];  % np. ~0.5s i ~1.0s (dwie sekwencje)
trigger_index = 1;

% Padding sygnału (podobnie jak wcześniej, z przodu i tyłu N_long zer)
x_pad = [zeros(N_long,1); x; zeros(N_long,1)];
N_pad = length(x_pad);

while pos + L_long <= N_pad  % dopóki możemy pobrać pełny długi blok
    if trigger_index <= length(trigger_points) ...
       && output_count >= trigger_points(trigger_index)
        % Czas na przełączenie -> użyj okna start + sekwencja krótkich
        % Blok z oknem start
        frame = x_pad(pos+1 : pos+L_long);
        X_all{end+1} = mdct_block(frame, w_start);
        win_types{end+1} = 'start';
        pos = pos + N_long;         % przesuwamy o 128 nowych próbek (długie okno)
        output_count = output_count + N_long;
        prev_mode = 'short';        % wejdziemy w tryb krótkich
        % 4 krótkie bloki
        for i = 1:4
            frame_s = x_pad(pos+1 : pos+L_short);
            X_all{end+1} = mdct_block(frame_s, w_short);
            win_types{end+1} = 'short';
            pos = pos + N_short;   % przesuwamy o 32 próbki
            output_count = output_count + N_short;
        end
        % Blok z oknem stop
        frame_end = x_pad(pos+1 : pos+L_long);
        X_all{end+1} = mdct_block(frame_end, w_stop);
        win_types{end+1} = 'stop';
        pos = pos + N_long;
        output_count = output_count + N_long;
        prev_mode = 'long';        % wracamy do trybu długich okien
        trigger_index = trigger_index + 1;  % zużyliśmy jeden trigger
    else
        % Brak przełączenia -> normalny blok (long or short w zależności od trybu)
        if strcmp(prev_mode, 'long')  % normalny długi blok
            frame = x_pad(pos+1 : pos+L_long);
            X_all{end+1} = mdct_block(frame, w_long);
            win_types{end+1} = 'long';
            pos = pos + N_long;
            output_count = output_count + N_long;
            prev_mode = 'long';
        else
            % Choć w tej logice prev_mode 'short' wystąpi tylko wewnątrz sekwencji, 
            % dla ogólności umożliwiamy obsługę ciągłych short (gdyby było trzeba)
            frame_s = x_pad(pos+1 : pos+L_short);
            X_all{end+1} = mdct_block(frame_s, w_short);
            win_types{end+1} = 'short';
            pos = pos + N_short;
            output_count = output_count + N_short;
            prev_mode = 'short';
        end
    end
    if pos + L_long > N_pad
        break;  % wyjście gdy niepełny blok pozostał
    end
end

% SYNTEZA: przejście przez wszystkie bloki X_all z odpowiednimi oknami
x_rec_padded = zeros(length(x_pad), 1);
prev_overlap_long = zeros(N_long, 1);
prev_overlap_short = zeros(N_short, 1);
pos_out = 0;
for b = 1:length(X_all)
    X_block = X_all{b};
    wtype = win_types{b};
    if strcmp(wtype, 'long') || strcmp(wtype, 'start') || strcmp(wtype, 'stop')
        Ncurr = N_long; Lcurr = L_long;
        if strcmp(wtype, 'long'), win_curr = w_long;
        elseif strcmp(wtype, 'start'), win_curr = w_start;
        else, win_curr = w_stop;
        end
        frame_rec = imdct_block(X_block, win_curr);
        x_out = prev_overlap_long + frame_rec(1:Ncurr);
        x_rec_padded(pos_out+1 : pos_out+Ncurr) = x_out;
        prev_overlap_long = frame_rec(Ncurr+1:end);
        prev_overlap_short = zeros(N_short, 1);
        pos_out = pos_out + Ncurr;
    else
        Ncurr = N_short; Lcurr = L_short;
        win_curr = w_short;
        frame_rec = imdct_block(X_block, win_curr);
        x_out = prev_overlap_short + frame_rec(1:Ncurr);
        x_rec_padded(pos_out+1 : pos_out+Ncurr) = x_out;
        prev_overlap_short = frame_rec(Ncurr+1:end);
        pos_out = pos_out + Ncurr;
        if b < length(X_all) && strcmp(win_types{b+1}, 'stop')
            prev_overlap_long = [prev_overlap_short; zeros(N_long-N_short,1)];
            prev_overlap_short = zeros(N_short, 1);
        end
    end
end

x_rec_switch = x_rec_padded(N_long+1 : end-N_long);
err_switch = x - x_rec_switch(1:Nx);
fprintf('III Max error po przełączaniu okien: %.3g\n', max(abs(err_switch)));


function X_blocks = mdct_analysis(x, N)
    L = 2*N;
    Nx = length(x);
    % Padding: dopełnij z tyłu tyle zer, by (Nx + pad) - N było podzielne przez N
    pad = mod(-(Nx - N), N);
    x_pad = [x; zeros(pad + N,1)]; % dodatkowe N zer do końca (dla ostatniego bloku)
    Nx_pad = length(x_pad);
    num_blocks = floor((Nx_pad - L)/N) + 1;
    n = (0:L-1)';
    win = sin((n + 0.5) * pi / (2*L));
    X_blocks = zeros(N, num_blocks);
    for b = 1:num_blocks
        start_idx = (b-1)*N + 1;
        frame = x_pad(start_idx : start_idx + L - 1);
        frame_win = frame .* win;
        Xk = zeros(N, 1);
        for k = 0:N-1
            n_vec = 0:L-1;
            Xk(k+1) = sum(frame_win .* cos((2*n_vec + 1 + N) * (2*k + 1) * pi/(4*N))');
        end
        X_blocks(:, b) = Xk;
    end
end

function x_rec = imdct_synthesis(X_blocks, N)
    L = 2*N;
    num_blocks = size(X_blocks, 2);
    n = (0:L-1)';
    win = sin((n + 0.5) * pi / (2*L));
    x_rec_padded = zeros((num_blocks-1)*N + L, 1);
    prev_overlap = zeros(N, 1);
    for b = 1:num_blocks
        Xk = X_blocks(:, b);
        x_block = zeros(L, 1);
        for n_idx = 0:L-1
            x_block(n_idx+1) = sum(Xk .* cos((2*n_idx + 1 + N) * (2*(0:N-1)' + 1) * pi/(4*N)));
        end
        x_block = (2/N) * x_block;
        x_block = x_block .* win;
        x_out = prev_overlap + x_block(1:N);
        prev_overlap = x_block(N+1:end);
        start_idx = (b-1)*N + 1;
        x_rec_padded(start_idx : start_idx + N - 1) = x_out;
        if b == num_blocks
            x_rec_padded(start_idx+N : start_idx+2*N-1) = prev_overlap;
        end
    end
    x_rec = x_rec_padded(1:((num_blocks)*N));
end

% Funkcja pomocnicza do MDCT dla pojedynczego bloku o zmiennym oknie:
function X = mdct_block(frame, win)
    L = length(win);
    N = L/2;
    n = 0:L-1;
    X = zeros(N, 1);
    frame_win = frame .* win;
    for k = 0:N-1
        X(k+1) = sum(frame_win .* cos((2*n + 1 + N) * (2*k + 1) * pi/(4*N))');
    end
end

% Funkcja pomocnicza do IMDCT dla pojedynczego bloku:
function frame = imdct_block(X, win)
    N = length(X);
    L = 2*N;
    n = 0:L-1;
    frame = zeros(L, 1);
    for nn = 0:L-1
        frame(nn+1) = sum(X .* cos((2*nn + 1 + N) * (2*(0:N-1)' + 1) * pi/(4*N)));
    end
    frame = (2/N) * frame;     % skalowanie
    frame = frame .* win;      % okno na wyjściu
end