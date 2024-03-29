**Целью** эксперимента является определить, есть ли различие в весах аттеншена для разных аргументов глагола. Предполагается, что модель должна для каждой роли присваивать наибольшие веса в разных головах и слоях. 

**Методы**: Эксперимент проводился на модели `sberbank-ai/ruBert-base`. Поскольку иногда слова разделялись на несколько токенов, было решено рассмотреть 3 варианта подсчета значения аттеншена между токенами одного слова: 
* считать среднее значение (mean);
* брать максимальное значение среди токенов (max);
* брать значение для первого токена (st).

В итоге получилось 9 вариантов, с разными вариантами подсчета аттеншена для глагола и аргумента.

Для каждого предложения были найдены значения на 12 слоях и в 12 головах. Затем данные усреднялись для каждого аргумента, после чего для каждой пары head-layer было найдено максимальное значение аттеншена и роль, которой это значение принадлежит. 

**Результаты**: 

Как видно из таблицы, для каждой роли есть как минимум одна head-layer пара, где значение аттеншена выше, чем у остальных ролей. Цветом показана величина каждого значения относительно других. Можно заметить, что четырем ролям: agent, patient, posessor и experiencer уделяется наибольшее внимание. 
