# VoteAGH - System Głosowania AGH

Aplikacja webowa do przeprowadzania ankiet i głosowań dla społeczności AGH, zintegrowana z Azure AD.

## Architektura

Aplikacja składa się z dwóch części:

- **Backend**: Spring Boot 3.2 + Java 21
- **Frontend**: Angular 17 + TypeScript

## Szybki Start

### Pobierz node.js
https://nodejs.org/en/download

### Konfiguracja JAVA_HOME

- Kliknij prawym przyciskiem na Ten komputer → Właściwości.
- Wybierz Zaawansowane ustawienia systemu.
- Kliknij Zmienne środowiskowe.
- W sekcji Zmienne systemowe kliknij Nowa...
- Nazwa zmiennej: JAVA_HOME
- Wartość zmiennej: wklej ścieżkę do katalogu JDK(lokalizacja Javy)
- Zatwierdź OK

### W Zmienne środowiskowe znajdź zmienną Path i dodaj na jej końcu:
%JAVA_HOME%\bin

### Backend

```bash
cd backend
.\mvn spring-boot:run
```

Backend będzie dostępny pod adresem: http://localhost:8080

### Frontend

```bash
cd frontend
npm install
npm start
```

Frontend będzie dostępny pod adresem: http://localhost:4200

## Wymagania Systemowe

### Backend
- Java 21
- Maven 3.6+
- H2 DB

### Frontend
- Node.js 18+
- npm 

## Konfiguracja Azure AD

1. Utwórz aplikację w Azure AD
2. Skonfiguruj redirect URI: `http://localhost:8080/login/oauth2/code/azure`
3. Ustaw zmienne środowiskowe:

## Funkcjonalności

### Dla Użytkowników
- ✅ Przeglądanie listy ankiet
- ✅ Głosowanie w ankietach
- ✅ Tworzenie nowych ankiet
- ✅ Wyświetlanie wyników w czasie rzeczywistym
- ✅ Logowanie przez Azure AD

### Reguły
- ✅ Jeden głos na użytkownika na ankietę
- ✅ Ignorowanie i nie przyjmowanie głosów po zamknięciu ankiety
- ✅ Obsługa ankiet z wielokrotnym wyborem
- ✅ Walidacja danych wejściowych


## API Endpoints

| Metoda | Endpoint | Opis |
|--------|----------|------|
| GET | `/api/polls` | Lista wszystkich ankiet |
| GET | `/api/polls/{id}` | Szczegóły ankiety |
| POST | `/api/polls` | Tworzenie nowej ankiety |
| GET | `/api/polls/my` | Moje ankiety |
| POST | `/api/polls/{id}/votes` | Głosowanie |
| GET | `/api/polls/{id}/votes` | Wyniki głosowania |

