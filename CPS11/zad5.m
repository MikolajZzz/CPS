%rng( 0 );
%x4 = randi( [1 5], 1, 10 )
% 5     5     1     5     4     1     2     3     5     5

clear; close all; clc;

%% 1. Wczytanie danych z pliku lab11.txt
% Plik powinien znajdować się w bieżącym katalogu MATLAB‐a.
filename = 'lab11.txt';
fid = fopen(filename, 'r');
x5 = fscanf(fid, '%f');   % wczytujemy wszystkie liczby (jako wektor kolumnowy)
fclose(fid);

%% 2. Obliczenie : unikalne symbole i ich częstości
[symb, prawd] = sortuj(x5);
% symb – wektor unikalnych wartości w x5
% prawd – wektor liczebności (liczniki każdego symbolu)

%% 3. Budowa drzewa Huffmana
tr = drzewo(symb, prawd);
% tr – struktura opisująca korzeń drzewa Huffmana

%% 4. Utworzenie tablicy kodera: lista struktur {symbol, bits}
kod = tablicaKodera([], tr);
% kod to tablica struktur z polami:
%   kod(i).symbol – wartość symbolu
%   kod(i).bits   – odpowiadający ciąg bitów 

%% 5. Zakodowanie całej sekwencji x5
% Najpierw zbudujemy mapę: od wartości symbolu do ciągu bitów
mapa = containers.Map('KeyType','double','ValueType','char');
for i = 1:length(kod)
    mapa(kod(i).symbol) = kod(i).bits;
end

% Teraz konkatenacja ciągów bitów
strumienBits = '';  % pusty łańcuch znaków
for i = 1:length(x5)
    bity = mapa(x5(i));
    strumienBits = [strumienBits, bity];
end

fprintf('Długość zakodowanego strumienia bitów: %d\n', length(strumienBits));

%% 6. Losowa zmiana jednego bitu
N = length(strumienBits);
idx_flip = randi(20);             % losowy indeks od 1 do N
orygBit = strumienBits(idx_flip);
% Odwrócenie bitu: '0' <-> '1' 
if orygBit == '0' %%Zmienić na 1 aby sprawdzić bez bitu losowego
    nowyBit = '1';
else
    nowyBit = '0';
end
strumienBits_zmieniony = strumienBits;
strumienBits_zmieniony(idx_flip) = nowyBit;


%% 7. Dekodowanie zmienionego strumienia
% Przy dekodowaniu przechodzimy drzewo Huffmana: '1'->lewy potomek, '0'->prawy potomek
odebrane = zeros(size(x5));
pos_wr  = 1;
node    = tr;
i_bit   = 1;
while i_bit <= length(strumienBits_zmieniony)
    if isempty(node.symbol)
        if strumienBits_zmieniony(i_bit) == '1'
            node = node.left;
        else
            node = node.right;
        end
        i_bit = i_bit + 1;
    else
        odebrane(pos_wr) = node.symbol;
        pos_wr = pos_wr + 1;
        node = tr;
    end
    if pos_wr > length(x5)
        break;
    end
end

% Dopisanie ostatniego symbolu
if pos_wr <= length(x5) && ~isempty(node.symbol)
    odebrane(pos_wr) = node.symbol;
    pos_wr = pos_wr + 1;
end

%% 8. Porównanie zdekodowanej sekwencji z oryginałem
idx_bledne = find(odebrane ~= x5);
num_blednych = length(idx_bledne);
fprintf('Liczba błędnie odtworzonych symboli: %d na %d.\n', num_blednych, length(x5));
if num_blednych > 0
    fprintf('Pozycje błędne: ');
    fprintf('%d ', idx_bledne);
    fprintf('\n');
end

%% (Fakultatywnie) Wyświetlenie kilku pierwszych symboli oryginału i po dekodowaniu
disp('Oryginał (pierwsze 20 symboli):');
disp(x5(1:min(20,end))');
disp('Dekodowane (pierwsze 20 symboli) ze zmianą bitu:');
disp(odebrane(1:min(20,end))');



% -------------------------------------------------------------------------
% Poniżej: lokalne funkcje sortuj, drzewo, tablicaKodera
% -------------------------------------------------------------------------

function [symb, prawd] = sortuj(x)
    % Funkcja przyjmuje wektor x (dowolne liczby).
    % Zwraca:
    %   symb – unikalne symbole (posortowane rosnąco),
    %   prawd – odpowiadające im liczebności w x.
    symb = unique(x);
    prawd = zeros(size(symb));
    for n = 1:length(prawd)
        prawd(n) = sum(x == symb(n));
    end
end

function tr = drzewo(symb, prawd)
    % Buduje drzewo Huffmana. Na wejściu:
    %   symb  – wektor unikalnych symboli,
    %   prawd – odpowiadające im (nieznormalizowane) prawdopodobieństwa/liczebności.
    % Zwraca tr – pojedynczą strukturę drzewa (korzeń).
    
    % Na początku tworzymy listę węzłów liści:
    for n = 1:length(prawd)
        node.pr = prawd(n);
        node.left = [];
        node.right = [];
        node.symbol = symb(n);
        tr(n) = node;
    end
    
    % Iteracyjnie łączymy dwa najmniejsze węzły:
    while length(tr) > 1
        pr = cat(1, tr.pr);
        [~, idx] = sort(pr, 'ascend');
        tr = tr(idx);                    % posortuj według rosnąco po pr
        trNew.pr = tr(1).pr + tr(2).pr;   % suma prawdopodobieństw
        trNew.left = tr(1);              % lewy poddrzewo
        trNew.right = tr(2);             % prawy poddrzewo
        trNew.symbol = [];               % wewnętrzny węzeł nie ma symbolu
        tr = [tr(3:end), trNew];         % usuń pierwsze dwa i dodaj nowy
    end
    % Na wyjściu tr jest macierzą 1x1: jedyna struktura to korzeń drzewa
end

function kod = tablicaKodera(kod, tr)
    % Funkcja rekurencyjnie przechodzi drzewo Huffmana i tworzy tablicę
    % struktur z polami: symbol oraz bits.
    % Wejście:
    %   kod – początkowo puste [] (przekazywane w wywołaniach rekurencyjnych),
    %   tr  – bieżący poddrzewo (początkowo korzeń).
    %
    % Wyjście:
    %   kod – tablica struktur ze wszystkimi liśćmi drzewa.
    
    if ~isempty(tr.symbol)
        % Jesteśmy w liściu: dodajemy do kodera
        idx = length(kod) + 1;
        kod(idx).symbol = tr.symbol;
        kod(idx).bits = '';  % na razie pusty ciąg – będzie wypełniany w drodze powrotnej
        return;
    end
    
    % Rekurencyjnie przejdź w lewo i w prawo
    kodleft  = tablicaKodera(kod, tr.left);
    kodright = tablicaKodera(kod, tr.right);
    
    % Wszystkie kody z lewego poddrzewa dostają na początku '1'
    for n = 1:length(kodleft)
        kodleft(n).bits = ['1', kodleft(n).bits];
    end
    % Wszystkie kody z prawego poddrzewa dostają na początku '0'
    for n = 1:length(kodright)
        kodright(n).bits = ['0', kodright(n).bits];
    end
    
    % Scal tablice z obu poddrzew
    kod = [kod, kodleft, kodright];
end
