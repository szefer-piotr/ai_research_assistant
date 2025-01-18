# Scenariusz I: Wyszukiwanie artykułów 

## Cel:

Przetestowanie, czy funkcjonalności asystenta w zakresie przeszukiwania podobnych artykułów w wyszukiwarkach są w zakresie możliwości ChataGPT.

## Opis scenariusza:
Wyszukiwarkę podobnych artykułów z internetu i możliwości wnioskowania na jego podstawie.
Wyszukiwanie podobnych zestawów danych z DRYAD i pracy z nimi, a kokretnie, wyciąganie informacji o przeprowadzonych analizach statystycznych

## Uwagi:
:warning: ChatGPT nie ma dostępu do pełnych wersji tekstu w pdf, myli się odnośnie typu analizy, np. statystyczn ana danych empirycznych vs. teoretyczne, modelowanie matematyczne. Jednak po załadowaniu PDFa dobrze sobie radzi.



## Notatki

ChatGPT4o i ChatGPT4o-mini mają możliwość dodania *instrukcji*. Podczas ich testowania dodałem nastęującą instrukcję do każdego promptu:

*“You are an expert in statistical analysis of ecological data with proficiency in R. Students will consult you regarding their data analysis needs and seek your assistance in conducting statistical analyses. Your role is to help them interpret and analyze their data, especially considering their limited statistical knowledge. Additionally, you will offer guidance, suggestions, and recommendations to address their queries effectively.”*

Symulacja mojego rozwiązania korzystając z ChataGPT (poprzez stronę a nie API):

Pierwszy scenariusz - zaczynam od szukania papera:

Wyszukaj paper naukowy, który porusza interesująca nas tematykę - (chatgpt ma możliwość użycia wyszukiwarki)

Po jego znalezieniu:

Porozmawiaj na jego temat - wymyśl jakieś pytania badawcze

Wyszukaj inne papery korzystając z arxiva (chatgpt ma możliwość użycia wyszukiwarki)

Wyszukaj inne zbiory danych korzystając z DRYAD (chatgpt ma możliwość użycia wyszukiwarki)



Proponuje abyś scenariusz wykonał dla 3 modeli:
- gpt4o-mini
- gpt4o
- o1

I zwrócił uwagę na ich odpowiedzi - czym się różnią? Może któreś są bardziej szczegółowe? Jak wygląda język - tehcniczny, ogólny? Jak obszerne są odpowiedzi?

Na pewno z punktu widzenia kosztów, wybierzemy na ten moment gpt4o-mini, ale chciałbym żebyś zobaczył:

jaki jest potencjał w droższych modelach

ile “tracisz” używając tańszego modelu

Proponuje te różnice udokumentowąc np. zdjęciami odpowiedzi na to samą zadaną instrukcję oraz opisać własnymi słowami jakie różnice dostrzegasz. To Ci się przysłuży w przyszłości, kiedy produkt się uda i będziesz zastanawiał się co dalej? Co można poprawić w rozwiązaniu? Rzucisz okiem na dokumentację i będziesz wiedział jak to wyglądało.

Na pewno też do tego wrócimy w trakcie spotkań/konsultacji, bo może się to nam przydać.