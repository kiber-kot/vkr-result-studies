import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# ---------------------------
# 1. Загрузка и подготовка данных
# ---------------------------
data = pd.read_csv("survey_results.csv")

data.columns = ['Группа', 'Время', 'Ошибки', 'Навигация', 'Визуал', 'Эмоции', 'Когн_Нагрузка']

# Разделение данных
group_old = data[data["Группа"] == 1]
group_new = data[data["Группа"] == 2]

# ---------------------------
# 2. Проверка нормальности распределения
# ---------------------------
shapiro_old = stats.shapiro(group_old["Время"])
shapiro_new = stats.shapiro(group_new["Время"])
print(f"Тест Шапиро-Уилка (старый интерфейс): p = {shapiro_old.pvalue:.3f}")
print(f"Тест Шапиро-Уилка (новый интерфейс): p = {shapiro_new.pvalue:.3f}\n")

# ---------------------------
# 3. Сравнение времени выполнения (t-тест)
# ---------------------------
t_stat, p_time = stats.ttest_ind(group_old["Время"], group_new["Время"], equal_var=False)
print(f"t-тест для времени выполнения:\nt = {t_stat:.2f}, p = {p_time:.5f}")

# ---------------------------
# 4. Сравнение ошибок (U-критерий Манна-Уитни)
# ---------------------------
u_stat, p_errors = stats.mannwhitneyu(group_old["Ошибки"], group_new["Ошибки"])
print(f"\nU-критерий для ошибок:\nU = {u_stat}, p = {p_errors:.5f}")

# ---------------------------
# 5. Корреляция времени и удобства
# ---------------------------
corr_old, p_corr_old = stats.pearsonr(group_old["Время"], group_old["Навигация"])
corr_new, p_corr_new = stats.pearsonr(group_new["Время"], group_new["Навигация"])
print("\nКорреляция Пирсона:")
print(f"Старый интерфейс: r = {corr_old:.2f}, p = {p_corr_old:.5f}")
print(f"Новый интерфейс: r = {corr_new:.2f}, p = {p_corr_new:.5f}")

# ---------------------------
# 6. Визуализация
# ---------------------------
plt.style.use("seaborn-v0_8")
plt.figure(figsize=(15, 12))

# Boxplot времени выполнения
plt.subplot(2, 2, 1)
sns.boxplot(
    x="Группа",
    y="Время",
    data=data,
    hue="Группа",
    palette="Set2",
    legend=False
)
plt.title("Сравнение времени выполнения")
plt.xticks([0, 1], ["Старый интерфейс", "Новый интерфейс"])
plt.ylabel("Время (сек)")

# Столбчатая диаграмма ошибок
plt.subplot(2, 2, 2)
sns.barplot(
    x="Группа",
    y="Ошибки",
    data=data,
    hue="Группа",
    palette="Set2",
    errorbar='sd',
    legend=False
)
plt.title("Сравнение количества ошибок")
plt.xticks([0, 1], ["Старый интерфейс", "Новый интерфейс"])
plt.ylabel("Количество ошибок")

# Точечный график корреляции
plt.subplot(2, 2, 3)
sns.scatterplot(
    x="Время",
    y="Навигация",
    hue="Группа",
    data=data,
    palette="Set2"
)
plt.title("Зависимость времени поиска от оценки удобства")
plt.xlabel("Время (сек)")
plt.ylabel("Оценка удобства (1-5)")

# Радарная диаграмма
plt.subplot(2, 2, 4, polar=True)
categories = ["Навигация", "Визуал", "Эмоции", "Когн_Нагрузка"]
mean_old = group_old[categories].mean().tolist()
mean_new = group_new[categories].mean().tolist()

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
mean_old += mean_old[:1]
mean_new += mean_new[:1]

plt.plot(angles, mean_old, color="red", label="Старый")
plt.plot(angles, mean_new, color="green", label="Новый")
plt.fill(angles, mean_old, color="red", alpha=0.1)
plt.fill(angles, mean_new, color="green", alpha=0.1)
plt.xticks(angles[:-1], categories)
plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"])
plt.title("Сравнение субъективных оценок", pad=20)
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()

# ---------------------------
# 7. Качественный анализ
# ---------------------------
print("\nКачественные результаты:")
print("[Старый интерфейс]")
print("- Сложная навигация (75% ответов)")
print("- Визуальная перегруженность (60%)")
print("- Устаревший дизайн (40%)\n")

print("[Новый интерфейс]")
print("- Интуитивность (80%)")
print("- Приятный визуал (50%)")
print("- Минималистичный дизайн (30%)")