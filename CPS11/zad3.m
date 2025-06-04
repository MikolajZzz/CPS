%% Plik główny: zadanie3_podpasmowe.m
% Kod do wykonania zadania 3 (część 1 i 2) – pierwsze 6 sekund sygnału
% „DontWorryBeHappy.wav”. Bez użycia dodatkowych bibliotek.

clear; close all; clc;

%% Parametry wczytania sygnału
[signal_full, fs] = audioread('DontWorryBeHappy.wav');
signal_full = signal_full(:,1);       % weźmy lewy kanał (mono)
N6s = 6 * fs;                         % liczba próbek odpowiadająca 6 s
x = signal_full(1:N6s);

%% Normalizacja do przedziału [-1,1]
x = x / max(abs(x));

%% Wariant 1: 8 podpasm, 6 bitów na każde podpasmo
M1 = 8;
q1 = 6;                               % 6 bitów → 2^6 poziomów
fprintf('Wariant 1: %d podpasm, %d bitów/pasmo\n', M1, q1);
[y1, bps1] = kodowanie_podpasmowe(x, M1, q1);
fprintf('  Średnia liczba bitów na próbkę (wariant 1): %.3f bpp\n', bps1);

%% Wariant 2: 32 podpasm, 6 bitów na każde podpasmo
M2 = 32;
q2 = 6;
fprintf('Wariant 2: %d podpasm, %d bitów/pasmo\n', M2, q2);
[y2, bps2] = kodowanie_podpasmowe(x, M2, q2);
fprintf('  Średnia liczba bitów na próbkę (wariant 2): %.3f bpp\n', bps2);

%% Wariant 3: 32 podpasm, zmienna liczba bitów na pierwsze 5 podpasm
M3 = 32;
qv = [8, 8, 7, 6, 4];                  % wektor długości 5; reszta zostanie wypełniona wartością 4
fprintf('Wariant 3: %d podpasm, zmienna liczba bitów [%s ...]\n', M3, num2str(qv));
[y3, bps3] = kodowanie_podpasmowe(x, M3, qv);
fprintf('  Średnia liczba bitów na próbkę (wariant 3): %.3f bpp\n', bps3);

%% Wyświetlenie wykresów czasowych (fragment sygnału oryginalnego i skompresowanych)
t = (0:length(x)-1) / fs;

figure('Name','Przebieg sygnału – oryginał i wyniki kompresji','Units','normalized','Position',[0.1 0.1 0.8 0.8]);
subplot(4,1,1);
plot(t, x, 'k');
title('Oryginalny sygnał (pierwsze 6 s)');
ylabel('Amplituda'); grid on;

subplot(4,1,2);
plot(t, y1, 'r');
title('Rekonstrukcja – Wariant 1 (8×6 bit)');
ylabel('Amplituda'); grid on;

subplot(4,1,3);
plot(t, y2, 'b');
title('Rekonstrukcja – Wariant 2 (32×6 bit)');
ylabel('Amplituda'); grid on;

subplot(4,1,4);
plot(t, y3, 'm');
title('Rekonstrukcja – Wariant 3 (32 zmienne bity)');
xlabel('Czas [s]');
ylabel('Amplituda'); grid on;

win   = 1024;
hop   = win/2;
nfft  = 1024;

figure('Name','Spektrogram sygnałów','Units','normalized','Position',[0.1 0.1 0.8 0.8]);

%% ----------------- 1) Spektrogram oryginalny ----------------- %%
subplot(4,1,1);
% Pobieramy macierz STFT
[Sx, Fx, Tx] = spectrogram(x, win, win-hop, nfft, fs);
% Przeliczamy amplitudy na dB:
Sx_db = 20*log10(abs(Sx) + eps);   % +eps by uniknąć log(0)
% Rysujemy mapę: w osiach T=tx, f=Fx/1000 (w kHz)
imagesc(Tx, Fx/1000, Sx_db);
axis xy;
ylim([0, fs/2/1000]);               % ograniczamy do 0–fs/2 w kHz
colorbar;
title('Spektrogram oryginalny');
xlabel('Czas [s]');
ylabel('Częstotliwość [kHz]');

%% ----------------- 2) Spektrogram – Wariant 1 (8×6 bit) ----------------- %%
subplot(4,1,2);
[Sy1, Fy1, Ty1] = spectrogram(y1, win, win-hop, nfft, fs);
Sy1_db = 20*log10(abs(Sy1) + eps);
imagesc(Ty1, Fy1/1000, Sy1_db);
axis xy;
ylim([0, fs/2/1000]);
colorbar;
title('Spektrogram – Wariant 1 (8×6 bit)');
xlabel('Czas [s]');
ylabel('Częstotliwość [kHz]');

%% ----------------- 3) Spektrogram – Wariant 2 (32×6 bit) ----------------- %%
subplot(4,1,3);
[Sy2, Fy2, Ty2] = spectrogram(y2, win, win-hop, nfft, fs);
Sy2_db = 20*log10(abs(Sy2) + eps);
imagesc(Ty2, Fy2/1000, Sy2_db);
axis xy;
ylim([0, fs/2/1000]);
colorbar;
title('Spektrogram – Wariant 2 (32×6 bit)');
xlabel('Czas [s]');
ylabel('Częstotliwość [kHz]');

%% ----------------- 4) Spektrogram – Wariant 3 (32 zmienne bity) ----------------- %%
subplot(4,1,4);
[Sy3, Fy3, Ty3] = spectrogram(y3, win, win-hop, nfft, fs);
Sy3_db = 20*log10(abs(Sy3) + eps);
imagesc(Ty3, Fy3/1000, Sy3_db);
axis xy;
ylim([0, fs/2/1000]);
colorbar;
title('Spektrogram – Wariant 3 (32 zmienne bity)');
xlabel('Czas [s]');
ylabel('Częstotliwość [kHz]');

%% Obliczenie stopnia kompresji
orig_bpp = 16;   % 16 bitów w pliku WAV (przed kompresją)
ratio1 = bps1 / orig_bpp;
ratio2 = bps2 / orig_bpp;
ratio3 = bps3 / orig_bpp;
fprintf('\nStopień kompresji (bps/orig_bpp):\n');
fprintf('  Wariant 1: %.3f (%.1f%% oryginału)\n', ratio1, ratio1*100);
fprintf('  Wariant 2: %.3f (%.1f%% oryginału)\n', ratio2, ratio2*100);
fprintf('  Wariant 3: %.3f (%.1f%% oryginału)\n', ratio3, ratio3*100);

%% Część 2: Dynamiczny przydział bitów
fprintf('\n***** Część 2: kodowanie adaptacyjne *****\n');
M_ad = 32;
frame_len = round(0.02 * fs);  % ramka 20 ms (~882 próbek)
[y_adapt, bps_adapt] = kodowanie_podpasmowe_adapt(x, M_ad, frame_len, fs);
fprintf('  Średnia liczba bitów na próbkę (adapt): %.3f bpp\n', bps_adapt);

%% Spektrogram – adaptacyjne kodowanie
figure('Name','Spektrogram – kodowanie adaptacyjne','Units','normalized','Position',[0.2 0.2 0.6 0.6]);

% Obliczenie STFT dla y_adapt
[S_ad, F_ad, T_ad] = spectrogram(y_adapt, win, win-hop, nfft, fs);

% Konwersja na dB (plus eps, żeby uniknąć -Inf)
S_ad_db = 20*log10(abs(S_ad) + eps);

% Rysowanie mapy
imagesc(T_ad, F_ad/1000, S_ad_db);
axis xy;
ylim([0, fs/2/1000]);
colorbar;
title('Spektrogram – kodowanie adaptacyjne (32 podpasm)');
xlabel('Czas [s]');
ylabel('Częstotliwość [kHz]');


%% ===== Funkcje pomocnicze poniżej ===== %%

%% funkcja: kodowanie_podpasmowe.m
function [y,bps] = kodowanie_podpasmowe(x,ch,q,sf)
    % x   : wektor sygnału mono
    % ch  : liczba podpasm
    % q   : liczba bitów na każde podpasmo (scalar lub wektor)
    % sf  : (opcjonalnie) wektor współczynników skalujących

    Nx = length(x);
    x = x(:)';
    if nargin < 4
        sf = [];
    end

    %% Przygotowanie filtrów i macierzy
    M = ch; 
    L = 16 * M;    
    MM = 2 * M;  
    Lp = L / MM;
    M2 = 2 * M;
    M3 = 3 * M;
    M4 = 4 * M;

    % Filtr prototypowy MPEG
    p = prototyp(L); 
    p = sqrt(M) * p;
    for nn = 1 : 2 : Lp-1
       p(nn*MM+1 : nn*MM+MM) = -p(nn*MM+1 : nn*MM+MM);
    end
    ap = reshape(p, 2*M, Lp)';
    ap = [ ap(1:Lp, 1:M), ap(1:Lp, M+1:2*M) ];

    % Macierze analizy/syntezy kosinusowej
    m = 0 : (2*M-1);
    for kk = 0 : M-1
       A(kk+1, 1:2*M) = 2 * cos((pi / M) * (kk + 0.5) .* (m - M/2));
       B(kk+1, 1:2*M) = 2 * cos((pi / M) * (kk + 0.5) .* (m + M/2));
    end

    %% Analiza podpasmowa
    K = floor(Nx / M);         % liczba pełnych grup M próbek
    sb = zeros(M, K);
    bx = zeros(1, L);

    for kk = 1 : K
        % Załaduj M próbek do bufora
        bx = [ x((kk-1)*M+1 : kk*M) fliplr(bx(1:L-M)) ];

        % Filtracja polifazowa
        pbx = p .* bx;                   % długość L
        a = reshape(pbx, M2, Lp);       % M2×Lp
        u = sum(a, 2);                   % M2×1
        sb(:, kk) = A * u;               % M×1
    end

    %% Skalowanie + kwantyzacja
    if isempty(sf)
        sf = 1 ./ max(abs(sb), [], 2)';   % wektor długości M
    elseif length(sf) == 1
        sf = sf .^ (1:M);
    elseif length(sf) < M
        sf(end+1 : M) = sf(end);
    elseif length(sf) > M
        sf = sf(1 : M);
    end
    sf_vec = sf(:)';
    sf_mat = diag(sf_vec);
    inv_sf_mat = diag(1 ./ sf_vec);

    if length(q) == 1
        Q = 2 ^ q;                      
        sbq = fix(Q * sf_mat * sb);     % (MxM) * (MxK) → M×K
        bits = sum( ceil(log2( max(sbq, [], 2) - min(sbq, [], 2) + 1 )) ) * size(sbq, 2);
        sbq = inv_sf_mat * sbq / Q;
    else
        qvec = 2 .^ q(:);               
        if length(qvec) < M
            qvec(end+1 : M) = qvec(end);
        elseif length(qvec) > M
            qvec = qvec(1 : M);
        end
        Qmat = diag(qvec');             
        inv_Qmat = diag(1 ./ qvec');
        sbq = fix(Qmat * sf_mat * sb);  % (MxM) * (MxK) → M×K
        bits = sum( ceil(log2( max(sbq, [], 2) - min(sbq, [], 2) + 1 )) ) * size(sbq, 2);
        sbq = inv_Qmat * inv_sf_mat * sbq;
    end
    bps = bits / Nx;

    %% Synteza podpasmowa
    y = zeros(Nx, 1);    % inicjalizacja długością Nx
    bv = zeros(1, 2 * L);

    for kk = 1 : K
        v = B' * sbq(:, kk);                       % M2×1
        bv = [ v' bv(1 : 2*L - MM) ];             % 1×(2L)
        abv = reshape(bv, M4, Lp);                % M4×Lp
        abv = abv';                               % Lp×M4
        abv = [ abv(:, 1:M), abv(:, M3+1 : M4) ];  % Lp×(2M)
        ys = sum(ap .* abv);                      % 1×(2M)
        y((kk-1)*M+1 : kk*M) = ys(1 : M);         % wstaw tylko pierwsze M próbek
    end

    %% Poniżej definicja funkcji prototyp w ramach kodowania_podpasmowe
    function p_out = prototyp(Lp_in)
        a0 = 1;
        a_vals(1) = -0.99998229;
        a_vals(2) = 0.99692250;
        a_vals(4) = 1 / sqrt(2);
        a_vals(6) = sqrt(1 - a_vals(2)^2);
        a_vals(7) = -sqrt(1 - a_vals(1)^2);
        Acoeff = -( a0/2 + a_vals(1) + a_vals(2) + a_vals(3) + a_vals(4) + a_vals(6) + a_vals(7) );
        a_vals(3) = Acoeff/2 - sqrt(0.5 - Acoeff^2/4);
        a_vals(5) = -sqrt(1 - a_vals(3)^2);
        n = 0 : Lp_in-1;
        p_out = a0 * ones(1, Lp_in);
        for k2 = 1 : 7
            p_out = p_out + 2 * a_vals(k2) * cos(2*pi*k2*n / Lp_in);
        end
        p_out = p_out / Lp_in;
        p_out(1) = 0;
    end
end


%% funkcja: kodowanie_podpasmowe_adapt.m
% Rozszerzenie o dynamiczny przydział bitów na każdą ramkę na podstawie energii w podpasmach.
function [y, avg_bps] = kodowanie_podpasmowe_adapt(x, ch, frame_len, fs)
    % x         : wektor sygnału mono
    % ch        : liczba podpasm (np. 32)
    % frame_len : długość ramki w próbkach (np. 0.02*fs ~882)
    % fs        : częstotliwość próbkowania (np. 44100)
    %
    % Wyjście:
    % y       : zrekonstruowany sygnał
    % avg_bps : średnia liczba bitów na próbkę w całym sygnale

    Nx = length(x);
    M = ch;
    L = 16 * M;
    MM = 2 * M;
    Lp = L / MM;
    M2 = 2 * M;
    M3 = 3 * M;
    M4 = 4 * M;

    %% Przygotowanie filtrów i macierzy
    p = prototyp(L);
    p = sqrt(M) * p;
    for nn = 1 : 2 : Lp-1
       p(nn*MM+1 : nn*MM+MM) = -p(nn*MM+1 : nn*MM+MM);
    end
    ap = reshape(p, 2*M, Lp)';
    ap = [ ap(1:Lp, 1:M), ap(1:Lp, M+1:2*M) ];
    m = 0 : (2*M-1);
    for kk = 0 : M-1
       A(kk+1, 1:2*M) = 2 * cos((pi / M) * (kk + 0.5) .* (m - M/2));
       B(kk+1, 1:2*M) = 2 * cos((pi / M) * (kk + 0.5) .* (m + M/2));
    end

    %% Inicjalizacja
    num_frames = floor(Nx / frame_len);
    y = zeros(Nx, 1);    % inicjalizacja długością Nx
    total_bits = 0;

    %% Petla po ramkach
    for fr = 1 : num_frames
        idx_start = (fr-1)*frame_len + 1;
        % wartość idx_end nie jest już używana w pełnym zakresie
        Kframe = floor(frame_len / M);
        frame = x(idx_start : idx_start + frame_len - 1)';    % 1×frame_len

        % ----- Analiza podpasmowa dla ramki -----
        sb_frame = zeros(M, Kframe);
        buf = zeros(1, L);
        for kk = 1 : Kframe
            seg = frame((kk-1)*M+1 : kk*M);
            buf = [ seg fliplr(buf(1 : L-M)) ];
            pb = p .* buf;                 % 1×L
            a_mat = reshape(pb, M2, Lp);  % M2×Lp
            u = sum(a_mat, 2);             % M2×1
            sb_frame(:, kk) = A * u;       % M×1
        end

        % ----- Dynamiczny przydział bitów -----
        E = sum(sb_frame .^ 2, 2);         % M×1: energia w każdym podpaśmie
        E_norm = E / sum(E);               % M×1
        bpf = M * 6;                       % budżet bitów na ramkę (jakby 6 bitów stałych)
        b_k = round(E_norm * bpf);         % M×1
        % Korekta sumy b_k do bpf:
        diff_bits = bpf - sum(b_k);
        if diff_bits > 0
            [~, idx_sort] = sort(E, 'descend');
            for ii = 1 : diff_bits
                b_k(idx_sort(ii)) = b_k(idx_sort(ii)) + 1;
            end
        elseif diff_bits < 0
            [~, idx_sort] = sort(E, 'ascend');
            for ii = 1 : abs(diff_bits)
                b_k(idx_sort(ii)) = max(1, b_k(idx_sort(ii)) - 1);
            end
        end
        b_k(b_k < 1) = 1;  % minimalnie 1 bit na pasmo

        % ----- Kwantyzacja z wektorem b_k -----
        sf = 1 ./ max(abs(sb_frame), [], 2)';  % M×1
        sf_mat = diag(sf);
        inv_sf_mat = diag(1 ./ sf);

        qvec = 2 .^ b_k;                       % M×1
        Qmat = diag(qvec);
        inv_Qmat = diag(1 ./ qvec);

        sbq_frame = fix(Qmat * sf_mat * sb_frame);  % (MxM)*(MxKframe) → M×Kframe
        bits_frame = sum( ceil(log2( max(sbq_frame, [], 2) - min(sbq_frame, [], 2) + 1 )) ) * Kframe;
        total_bits = total_bits + bits_frame;

        sbq_frame = inv_Qmat * inv_sf_mat * sbq_frame;
        % ------------------------------------------------

        % ----- Synteza podpasmowa dla ramki -----
        y_frame = zeros(1, Kframe * M);
        bufv = zeros(1, 2 * L);
        for kk = 1 : Kframe
            v = B' * sbq_frame(:, kk);            % M2×1
            bufv = [ v' bufv(1 : 2*L - MM) ];    % 1×(2L)
            abv = reshape(bufv, M4, Lp);         % M4×Lp
            abv = abv';                           % Lp×M4
            abv = [ abv(:, 1:M), abv(:, M3+1 : M4) ]; % Lp×(2M)
            ysamp = sum(ap .* abv);              % 1×(2M)
            y_frame((kk-1)*M+1 : kk*M) = ysamp(1 : M);  
        end

        % Przypisz do y tylko Kframe*M próbek
        y(idx_start : idx_start + Kframe*M - 1) = y_frame';
    end

    avg_bps = total_bits / Nx;

    %% Poniżej definicja funkcji prototyp w ramach kodowania_podpasmowe_adapt
    function p_out = prototyp(Lp_in)
        a0 = 1;
        a_vals(1) = -0.99998229;
        a_vals(2) = 0.99692250;
        a_vals(4) = 1 / sqrt(2);
        a_vals(6) = sqrt(1 - a_vals(2)^2);
        a_vals(7) = -sqrt(1 - a_vals(1)^2);
        Acoeff = -( a0/2 + a_vals(1) + a_vals(2) + a_vals(3) + a_vals(4) + a_vals(6) + a_vals(7) );
        a_vals(3) = Acoeff/2 - sqrt(0.5 - Acoeff^2/4);
        a_vals(5) = -sqrt(1 - a_vals(3)^2);
        n = 0 : Lp_in-1;
        p_out = a0 * ones(1, Lp_in);
        for k2 = 1 : 7
            p_out = p_out + 2 * a_vals(k2) * cos(2*pi*k2*n / Lp_in);
        end
        p_out = p_out / Lp_in;
        p_out(1) = 0;
    end
end
