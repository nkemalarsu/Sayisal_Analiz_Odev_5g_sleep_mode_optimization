import math
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# PROJE: 5G Baz İstasyonu Uyku Modu Optimizasyonu
# YÖNTEM: Newton-Raphson
# ==========================================================

# 1. PARAMETRELER (Fiziksel Modelden)
Ce = 20.0     # Enerji Tasarrufu Katsayısı
alpha = 0.5   # Gecikme Cezası Artış Hızı
beta = 10.0   # Baz Ceza Katsayısı

TOLERANCE = 1e-6
MAX_ITER = 100

# 2. FONKSİYONLAR
def total_cost_function(t):
    # f(t) = Gecikme Cezası - Enerji Tasarrufu
    return beta * math.exp(alpha * t) - (Ce * t)

def derivative_function(t):
    # f'(t) = beta * alpha * e^(alpha*t) - Ce
    return (beta * alpha * math.exp(alpha * t)) - Ce

def second_derivative_function(t):
    # f''(t)
    return beta * (alpha ** 2) * math.exp(alpha * t)

# 3. NEWTON-RAPHSON ALGORİTMASI
def solve_optimization():
    t = 1.0 # Başlangıç Tahmini (t0)
    
    print("\n" + "="*65)
    print(f"{'İterasyon':<10} {'t (Uyku Süresi)':<20} {'Eğim (Türev)':<20} {'Maliyet'}")
    print("="*65)
    
    history_t = []
    history_cost = []

    for i in range(MAX_ITER):
        f_val = total_cost_function(t)
        f_prime = derivative_function(t)
        f_double_prime = second_derivative_function(t)
        
        history_t.append(t)
        history_cost.append(f_val)
        
        print(f"{i:<10} {t:<20.6f} {f_prime:<20.6f} {f_val:.4f}")

        if abs(f_prime) < TOLERANCE:
            print("-" * 65)
            print(f">>> SONUÇ: {i}. adımda optimuma ulaşıldı.")
            return t, history_t, history_cost
            
        t_next = t - (f_prime / f_double_prime)
        t = t_next

    return t, history_t, history_cost

# 4. ÇALIŞTIRMA VE GRAFİK
optimal_t, hist_t, hist_cost = solve_optimization()

print(f"\n>>> OPTİMUM UYKU SÜRESİ: {optimal_t:.6f} ms")
print(f">>> MİNİMUM MALİYET: {total_cost_function(optimal_t):.6f}")

# Grafik Çizimi
t_values = np.linspace(0, 4, 100)
cost_values = [total_cost_function(val) for val in t_values]

plt.figure(figsize=(10, 6))
plt.plot(t_values, cost_values, label='Toplam Maliyet f(t)', color='blue', linewidth=2)
plt.scatter(hist_t, hist_cost, color='red', zorder=5, s=50, label='İterasyon Adımları')
plt.plot(optimal_t, total_cost_function(optimal_t), 'g*', markersize=20, label='Optimum Nokta')

plt.title('5G Uyku Modu Optimizasyonu (Newton-Raphson)')
plt.xlabel('Uyku Süresi (ms)')
plt.ylabel('Maliyet')
plt.grid(True, linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()