import csv

csv_path = 'features.csv'
images_path = 'images'
inverse_path = 'inverse'
profiles_path = 'profiles'
readme_path = 'results/README.md'

def create_readme(csv_path, images_path, inverse_path, profiles_path, readme_path):
    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')

        with open(readme_path, 'w', encoding='utf-8') as readme_file:
            readme_file.write("# Лабораторная работа №5. Выделение признаков символов\n\n")

            letters_to_process = ['А', 'Б', 'В', 'Г', 'Д']
            for row in reader:
                if row['letter'] in letters_to_process:
                    symbol = row['letter']

                    readme_file.write(f"## Символ - {symbol}\n\n")

                    readme_file.write("### Фото прямой буквы\n")
                    readme_file.write(f"![Прямая буква {symbol}](../{images_path}/{symbol}.png)\n\n")

                    readme_file.write("### Фото инвертированной буквы\n")
                    readme_file.write(f"![Инвертированная буква {symbol}](../{inverse_path}/{symbol}.png)\n\n")

                    readme_file.write("### Профили буквы\n")
                    readme_file.write(f"![Профиль X {symbol}](../{profiles_path}/{symbol}_profile_x.png)\n")
                    readme_file.write(f"![Профиль Y {symbol}](../{profiles_path}/{symbol}_profile_y.png)\n\n")

                    readme_file.write("### Признаки:\n")
                    readme_file.write(f"1. Вес I - {row['weight_I']}\n")
                    readme_file.write(f"2. Относительный вес I - {row['relative_weight_I']}\n")
                    readme_file.write(f"3. Вес II - {row['weight_II']}\n")
                    readme_file.write(f"4. Относительный вес II - {row['relative_weight_II']}\n")
                    readme_file.write(f"5. Вес III - {row['weight_III']}\n")
                    readme_file.write(f"6. Относительный вес III - {row['relative_weight_III']}\n")
                    readme_file.write(f"7. Вес IV - {row['weight_IV']}\n")
                    readme_file.write(f"8. Относительный вес IV - {row['relative_weight_IV']}\n")
                    readme_file.write(f"9. Общий вес - {row['total_weight']}\n")
                    readme_file.write(f"10. Относительный общий вес - {row['relative_total_weight']}\n")
                    readme_file.write(f"11. Центр тяжести Y - {row['center_y']}\n")
                    readme_file.write(f"12. Центр тяжести X - {row['center_x']}\n")
                    readme_file.write(f"13. Относительный центр тяжести Y - {row['relative_center_y']}\n")
                    readme_file.write(f"14. Относительный центр тяжести X - {row['relative_center_x']}\n")
                    readme_file.write(f"15. Момент инерции Y - {row['inertia_y']}\n")
                    readme_file.write(f"16. Момент инерции X - {row['inertia_x']}\n")
                    readme_file.write(f"17. Относительный момент инерции Y - {row['relative_inertia_y']}\n")
                    readme_file.write(f"18. Относительный момент инерции X - {row['relative_inertia_x']}\n\n")

            readme_file.write("## Вывод по работе\n")
            readme_file.write(
                "В ходе выполнения лабораторной работы были выделены признаки символов русского алфавита. "
                "Для каждого символа были рассчитаны вес, относительный вес, координаты центра тяжести, "
                "моменты инерции и их нормированные значения. Также были построены профили X и Y для каждого символа. "
                "Полученные данные могут быть использованы для дальнейшего анализа и классификации символов.\n")


create_readme(csv_path, images_path, inverse_path, profiles_path, readme_path)

print(f"README.md успешно создан: {readme_path}")