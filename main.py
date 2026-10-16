import re
import random
import math
import numpy as np
from collections import Counter, defaultdict
# 1. ЧТЕНИЕ И ПРЕДОБРАБОТКА ТЕКСТА
with open('metro-2033.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
# Приводим к нижнему регистру
raw_text = raw_text.lower()

# Решение по знакам препинания:
# Я ВЫБРАЛ УДАЛИТЬ знаки препинания и оставить только слова.
# Обоснование:
#  Если сделать знаки отдельными токенами, словарь разрастётся,а биграммная модель станет очень разреженной

# Регулярное выражение: слова (включая слова с дефисом и ё)
tokens = re.findall(r'[а-яёa-z0-9]+(?:-[а-яёa-z0-9]+)*', raw_text)

print(f"Всего токенов: {len(tokens)}")
print(f"Размер словаря: {len(set(tokens))}")
print(f"\n20 самых частых слов:")
freq = Counter(tokens)
for word, count in freq.most_common(20):
    print(f"  {word:15s} {count:6d}")
# 2. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ЧАСТИ
split_idx = int(len(tokens) * 0.9)
train_tokens = tokens[:split_idx]
test_tokens = tokens[split_idx:]

print(f"\nОбучающая выборка: {len(train_tokens)} токенов")
print(f"Тестовая выборка:  {len(test_tokens)} токенов")
class BigramLM:
    """Биграммная языковая модель со сглаживанием Лапласа."""

    def __init__(self, alpha=0.0):
        self.alpha = alpha          # параметр сглаживания
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()
        self.vocab_size = 0

    def fit(self, tokens):
        """Подсчёт частот по обучающей выборке."""
        self.unigram_counts = Counter(tokens)
        self.bigram_counts = defaultdict(Counter)
        for i in range(len(tokens) - 1):
            self.bigram_counts[tokens[i]][tokens[i+1]] += 1
        self.vocab = set(tokens)
        self.vocab_size = len(self.vocab)

    def prob(self, v, w):
        """P(w | v) со сглаживанием Лапласа."""
        c_vw = self.bigram_counts[v][w]
        c_v = self.unigram_counts[v]
        if self.alpha == 0:
            if c_v == 0:
                return 0.0
            return c_vw / c_v
        else:
            return (c_vw + self.alpha) / (c_v + self.alpha * self.vocab_size)

    def generate(self, start_word, length=30, temperature=1.0):
        """Генерация текста заданной длины от стартового слова."""
        if start_word not in self.vocab:
            return None
        result = [start_word]
        current = start_word
        for _ in range(length - 1):
            # Собираем вероятности для всех слов словаря
            probs = np.array([self.prob(current, w) for w in self.vocab])
            # Применяем температуру: P^(1/T), затем нормируем
            if temperature != 1.0:
                probs = np.power(probs + 1e-12, 1.0 / temperature)
                probs = probs / probs.sum()
            else:
                if probs.sum() == 0:
                    break
                probs = probs / probs.sum()
            next_word = np.random.choice(list(self.vocab), p=probs)
            result.append(next_word)
            current = next_word
        return ' '.join(result)

    def logprob(self, text_tokens):
        """Суммарный log P текста: Σ log P(w_t | w_{t-1})."""
        lp = 0.0
        for i in range(1, len(text_tokens)):
            p = self.prob(text_tokens[i-1], text_tokens[i])
            if p == 0:
                return float('-inf')
            lp += math.log(p)
        return lp

    def perplexity(self, tokens):
        """Перплексия на последовательности токенов."""
        lp = 0.0
        n = 0
        for i in range(1, len(tokens)):
            p = self.prob(tokens[i-1], tokens[i])
            if p == 0:
                return float('inf')
            lp += math.log(p)
            n += 1
        if n == 0:
            return float('inf')
        return math.exp(-lp / n)

#ОБУЧЕНИЕ МОДЕЛИ (без сглаживания)
model = BigramLM(alpha=0.0)
model.fit(train_tokens)
print("ГЕНЕРАЦИЯ (T=1.0, без сглаживания)")
print("="*60)
random.seed(42)
np.random.seed(42)

# Берём 5 разных стартовых слов из частых
start_words = ['артем', 'станция', 'туннель', 'черные', 'метро']
for start in start_words:
    text = model.generate(start, length=30, temperature=1.0)
    print(f"\n[{start}]: {text}")

# 6. ГЕНЕРАЦИЯ С РАЗНОЙ ТЕМПЕРАТУРОЙ
print("ВЛИЯНИЕ ТЕМПЕРАТУРЫ (стартовое слово: 'артем')")
print("="*60)

for T in [0.5, 1.0, 2.0]:
    print(f"\n--- T = {T} ---")
    for i in range(3):
        text = model.generate('артем', length=30, temperature=T)
        print(f"  {text}")



# 7. ПРАВДОПОДОБИЕ ТЕКСТА
print("ПРАВДОПОДОБИЕ: СВЯЗНОЕ vs ПЕРЕМЕШАННОЕ")
print("="*60)

# Берём связное предложение из тестовой части
# Найдём предложение длиной хотя бы 10 слов
def find_sentence(tokens, min_len=10):
    """Ищем подряд идущие токены (имитация предложения)."""
    # Возьмём просто 15 подряд идущих слов из теста
    return tokens[100:115]

coherent = find_sentence(test_tokens, 15)
shuffled = coherent.copy()
random.shuffle(shuffled)

print(f"Связный фрагмент:  {' '.join(coherent)}")
print(f"Перемешанный:      {' '.join(shuffled)}")

# Без сглаживания
model_no_smooth = BigramLM(alpha=0.0)
model_no_smooth.fit(train_tokens)
lp_coh = model_no_smooth.logprob(coherent)
lp_shuf = model_no_smooth.logprob(shuffled)
print(f"\nБЕЗ СГЛАЖИВАНИЯ:")
print(f"  log P(связный)    = {lp_coh}")
print(f"  log P(перемешанный) = {lp_shuf}")

# СО СГЛАЖИВАНИЕМ
model_smooth = BigramLM(alpha=0.1)
model_smooth.fit(train_tokens)
lp_coh_s = model_smooth.logprob(coherent)
lp_shuf_s = model_smooth.logprob(shuffled)
print(f"\nСО СГЛАЖИВАНИЕМ (α=0.1):")
print(f"  log P(связный)    = {lp_coh_s:.4f}")
print(f"  log P(перемешанный) = {lp_shuf_s:.4f}")
print(f"  Разница:           {lp_coh_s - lp_shuf_s:.4f}")


# 8. ОЦЕНКА: ПЕРПЛЕКСИЯ НА ТЕСТЕ
print("\n" + "="*60)
print("ПЕРПЛЕКСИЯ НА ТЕСТОВОЙ ВЫБОРКЕ")

print(f"{'α':>8s} | {'Перплексия':>12s}")
print("-" * 25)

for alpha in [1.0, 0.1, 0.01]:
    m = BigramLM(alpha=alpha)
    m.fit(train_tokens)
    ppl = m.perplexity(test_tokens)
    print(f"{alpha:8.3f} | {ppl:12.2f}")

print("\nГотово!")