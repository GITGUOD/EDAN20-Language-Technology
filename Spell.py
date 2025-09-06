"""Spelling Corrector in Python 3; see http://norvig.com/spell-correct.html

Copyright (c) 2007-2016 Peter Norvig
MIT license: www.opensource.org/licenses/mit-license.php
"""

################ Spelling Corrector 

import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower()) # Använder regex biblioteket, använder regex för att plocka ut alla ord

WORDS = Counter(words(open('big.txt').read())) # Öppnat big.txt filen och läser in den och letar efter alla ord samt räknar de

def P(word, N=sum(WORDS.values())): # Metod som tar in 2 parameter, den första är ett ord och det andra har default värdet att vi  tar in alla ord som värde och summerar de
    "Probability of `word`."
    return WORDS[word] / N # Beräknar sedan sannolikheten med den räknade summan av orden från texten och dividerar med antalet ord som finns i texten

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P) # Använder max funktionaliteten för att beräkna med sannolika kandidaten med metoden candidate för ordet som ska rättas till med hjälp av P funktionen ovanför som beräknar sannolikheten och den andra parametern tar emot en hel lista av ord

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word]) # Om det finns ord i listan eller om

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS) # Returnerar en set av ord som finns i WORDS.listan, set innehåller alltid unika värden

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz' # Alla bokstäver i alfabetet, där i med "Letters"
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)] # Delar upp ordet med hjälp av en for loop som går igenom hela ordet och delar upp det i två delar, vänster del och höger del
    deletes    = [L + R[1:]               for L, R in splits if R] # Tar bort en bokstav genom att låta höger delen av ordet börja på 1:, resterande bokstäver sätts in, vi skippar alltså indexet 0 där bokstaven vi vill ta bort finns
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1] # Tar den vänstra ordet L, lägger till den andra ordet på höger sida R[1], lägger till den första ordet på höger sida R[0] och lägger till resten av höger sida R[2:]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters] # Tar vänster del L, lägger till en bokstav C + resterande höger del R[1:], där vi byter ut en bokstav vid indexet R[0:1]
    inserts    = [L + c + R               for L, R in splits for c in letters] # Tar vänster del L, lägger till en bokstav + ordet åt höger sida R
    return set(deletes + transposes + replaces + inserts) # Returnerar en hel set av det 'deletes', 'transposes', 'replaces' och 'inserts'

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1)) # För ett ord som är 2 bokstäver felstavande/redigeringar från det rättstavade ordet, kör edits1 2 gånger eftersom den kollar först alla kandidater för om ordet skulle vara 1 redigering felstavat, sedan kör den igen för att kontrollera alla kandidater som är en redigering borta från edits1 igen.

################ Test Code 

def unit_tests():
    assert correction('speling') == 'spelling'              # insert # Testar olika felstavningar och hur programmet rättar de
    assert correction('korrectud') == 'corrected'           # replace 2
    assert correction('bycycle') == 'bicycle'               # replace
    assert correction('inconvient') == 'inconvenient'       # insert 2
    assert correction('arrainged') == 'arranged'            # delete
    assert correction('peotry') =='poetry'                  # transpose
    assert correction('peotryy') =='poetry'                 # transpose + delete
    assert correction('word') == 'word'                     # known
    assert correction('quintessential') == 'quintessential' # unknown
    assert words('This is a TEST.') == ['this', 'is', 'a', 'test']
    assert Counter(words('This is a test. 123; A TEST this is.')) == (
           Counter({'123': 1, 'a': 2, 'is': 2, 'test': 2, 'this': 2}))
    assert len(WORDS) == 32198, "%d unique words found" % len(WORDS)
    assert sum(WORDS.values()) == 1115585, "%d total number of words found" % sum(WORDS.values())
    assert WORDS.most_common(10) == [
        ('the', 79809), 
        ('of', 40024), 
        ('and', 38312), 
        ('to', 28765), 
        ('in', 22023), 
        ('a', 21124), 
        ('that', 12512), 
        ('he', 12401), 
        ('was', 11410), 
        ('it', 10681)
    ], "%s" % repr(WORDS.most_common(10)) # Kollar de 10 vanligaste orden i texten
    assert WORDS['the'] == 79809 # Kollar att 'the' stämmer med antalet 79809 i texten
    assert P('quintessential') == 0
    assert 0.07 < P('the') < 0.08
    return 'unit_tests pass'

# Kollar så att allting stämmer med texten-filen Big.txt

def spelltest(tests, verbose=False):
    "Run correction(wrong) on all (right, wrong) pairs; report results."
    import time
    start = time.time()
    good, unknown = 0, 0
    n = len(tests)
    for right, wrong in tests:
        w = correction(wrong)
        good += (w == right)
        if w != right:
            unknown += (right not in WORDS)
            if verbose:
                print('correction({}) => {} ({}); expected {} ({})'
                      .format(wrong, w, WORDS[w], right, WORDS[right]))
    dt = time.time() - start
    print('{:.0%} of {} correct ({:.0%} unknown) at {:.0f} words per second '
          .format(good / n, n, unknown / n, n / dt))
# Tar tid och beräknar hur många ord som är rätta och felaktiga, samt ord som inte finns med, där i med unknown
def Testset(lines):
    "Parse 'right: wrong1 wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs."
    return [(right, wrong)
            for (right, wrongs) in (line.split(':') for line in lines)
            for wrong in wrongs.split()]

if __name__ == '__main__':
    print(unit_tests())
    spelltest(Testset(open('spell-testset1.txt')))
    spelltest(Testset(open('spell-testset2.txt')))

# Huvud main programmet som kör metoderna samt printar ut unit testerna


# Teorin

# Kallelsen av metoden correction(w) utnyttjar en sannolikhetsmodell för att hitta den mest sannolika ordet som w kan vara en felstavning av.
# Modellen är baserad på sannolikhet P(W|C) eftersom vi maskinen kan veta exakt om t.ex 'lates' ska vara rättstavad till late eller latest eller lattes.
# Därför byggs modellen på att kalla alla möjliga kandidater via formen: argmaxc e candidates P(C) P(W|C) / P(W) och om man förenklar formeln eftersom sannolikheten är samma för
# Varje kandidat P(W) så får iv istället formen argmaxc e candidate P(C) P(W|C)
# argmax står för valet av den högst sannolika kandidaten kombineraet med all information modellen får
# c e candidates står för vilka korrektioner som kandidaten eller ordet då ska välja
# P(c) är sannolikheten ordet kommer upp vanligtvis i en engelsk text. Teorin förklarar här att t.ex
# 'the' utgör cirka 7% av alla engelska texter, så P(the) = 0.07
# P(W|C) är felmodellen som beräknar sannolikheten att rätta ordet är korrekt. T.ex P(teh|the) är högt men P(theeexyz|the) är låg
# Det vanligaste frågan är varför vi inte räknar P(C| W) direkt. Men svaret enligt sidan är ganska enkelt och det handlar om att 
# det är fler faktorer som kan påverka sannolikheten för att det rättstavade ordet ska komma fram.
# Nämligen hur vanligt ordet är i texten och hur felstavningen skedde så som att om man skulle skriva thew" av misstag
#  så kan maskinen tolka felstavningen som "thaw" eftersom ändringen a till e är en liten ändring. Dock så kanske ville den som skrev thew
# att thew skulle skrivas som the, då skulle detta också vara passande eftersom the är ett vanligt ord, därför är det mer passande och beräkna på det viset i beräkningen av att rätta ordet.

# I python så byter vi nämligen argmax till metoden max med ett argument som vi ser ovan.
# till kandidatmodellen så har vi en så kallad edit metod som editar, tar bort, swapar bokstäver, byter ut, lägger till i många omgångar. När den är klar, returneras en set av unika kombinationer och ord av kandidaten som har blivit felstavad.
# För ord som vanligt förekommande i ordboken så får vi en mindre unik lista jämfört med ord som inte är så pass kända om vi nu skulle stava fel.

# Om ordet är felstavat mer än liksom en bokstav iväg kan vi behöva en annan metod, i detta falet edits2 som generera en större set av möjligheter och är lite svårare och ta fram det rättstavade ordet eftersom inte så många sådana ord är kända ord.

#Språkmodellen
# Teorin är att vi kan beräkna sannolikheten of P(word) genom att räkna förekomsten av ordet i en viss text som är runt fem miljoner ord så som big.txt. 
# Detta gör vi i koden via metoden word som tar in text och översätter de till ord. Därefter används en counter för att räkna antalet förekomster ett ord finns i texten och
# P metoden beräknar sannolikheten baserad på hur många gånger det ordet förekommer.

# Fel modellen beräknar sannolikheten av att ordet är rätt baserat på hur felstavat ordet är. T.ex är ordet är endast en redigering ifrån har högre sannolikhet att bli beräknad som rättstavad utifrån en iteration av språkmodellen jämfört med ett ord som är två redigeringar ifrån.
# Sedan baseras sannolikheten dessitom om ordet är vanligt eller är känt i texten dessutom vilket metoderna correction(word) och candidates(word) har i uppgift att göra.

# Evaluation
# För att avgöra hur noggrann programmet är, används unit_tests för att se om programmet kan avgöra vad som är felstavat.


# Future work
# Fyra möjliga förbättringar:

    # 1. Utveckla modellen mot mer ord som inte är så pass kända så som att känna igen när ordet slutar på -able, -ity.
    # 2. Små fel skapar problem så som att att adres ska vara stavad som address och inte acres eftersom modellen använder sig av edit1 metoden istälelt för
    # 3. Begränsa redigeringar eftersom det inte finns så många ord som är felstavat mer än 2 redigeringar eftersom programmet för det mesta klarar majoriteten av fallen inom 2 redigeringar. Begränsningarna kan vara t.ex att byta ut vowels mot en annan voewel eller ersätta konsonanter som är lika så som 'c' till 's'
    # 4. Titta på meningen. T.ex ordet where, om man endast tittar på ordet så ger det inte så mycket om maskinen fick nu välja mellan where eller were. Om man dock hade en query så som "They where going" då hade det varit med större sannolikt att were var felstavat
    # 5. Optimera systemet via en kompilerat språk istället för en interpreted one. Cacha datorn så att vi inte behöver repetera så många gånger.