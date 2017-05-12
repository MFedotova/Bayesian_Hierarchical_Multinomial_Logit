library("rstan") # врубим на полную мощь Байеса!
library("shinystan") # и на полную громкость интерактивный анализ полученной цепи
library("knitr")
library("dplyr") # стратегия Разделяй - Властвуй - Соединяй
library("reshape2") # melt - cast
library("ggplot2") # графики
library("MCMCpack")
library("haven")
library("bayesm")
library("stringr")
library("tidyr")


rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Эта часть была взята у Бориса Борисовича отсюда: 
# https://bdemeshev.github.io/r_cycle/cycle_files/32_bayesian_hierarchical_logit.html

df <- read_spss("~/Downloads/conjoint_host_sim_dummy.sav")

glimpse(df) 
head(df) 

# Исходим из предположения, что каждому индивиду предлагалось одно и то же количество карточек. 
# Также предполагаем, что на каждой карточке было одно и то же количество альтернатив.
# Разрезаем исходную матрицу на две, y и X:
n_persons <- nrow(df) 
ncard_s <- 7 
n_alternatives <- 5

person_ids <- df$id
person_ids

df_X <- dplyr::select(df, -ends_with("select"))
df_y <- dplyr::select(df, ends_with("select"))
head(df_X)

# Превращаем матрицу X из широкой в длинную:
df_X_melted <- melt(df_X, id.vars = c("id", "version"))
head(df_X_melted)

# Разбиваем название переменной на три составляющих:
df_X_sep <- tidyr::separate(df_X_melted, variable, into = c("card", "alternative", "variable"), sep = "_")
head(df_X_sep) 

# Переделываем табличку с данными в список:
choice_data <- list()
for (person_no in 1:n_persons) {
    person_id <- person_ids[person_no]
    
    person_y <- unlist(df_y[person_no, ])
    names(person_y) <- NULL
    # unlist нужен чтобы превратить tbl_df размера (1 x n_cards) в простой вектор
    
    person_X_melted <- df_X_sep %>% dplyr::filter(id == person_id) %>% 
        dplyr::select(-id, -version)
    
    person_X_df <- dcast(person_X_melted, card + alternative  ~ variable, 
                         value.var = "value")
    person_X <- dplyr::select(person_X_df, -card, -alternative) %>% as.matrix()
    
    choice_data[[person_no]] <- list(y = person_y, X = person_X)
}

# Посмотрим на структуру матрицы y и X для второго индивида.
choice_data[[2]]
choice_data

# Теперь у нас для каждого человека своя матрица Х и вектор y.
# Тут мы перестаём воровать чужую обработку данных. Слезаем с удобной шеи и начинаем что-то делать сами :)

# Добавляем свободный коэффициент в матрицу X для каждого человека:
Intercept <- c(rep(1, 35))
new_data <- list() 
for (n in 1:296){
    z <- list()
    z$X = cbind(Intercept, choice_data[[n]]$X)
    z$y = choice_data[[n]]$y
    new_data[[n]] = z
}


# Заменяем в Payment и Personalization 2 на 0, чтобы получились дамми:
for (n in 1:296){
    for (i in 1:35){
        new_data[[n]]$X[i,7][new_data[[n]]$X[i,7] == 2] <- 0
        new_data[[n]]$X[i,8][new_data[[n]]$X[i,8] == 2] <- 0
    }
}

# Делим выборку на тренировочную и тестову:
set.seed(117)

indexes <- sample(1:7, 296, replace = TRUE)
train <- list()
test <- list()
for (i in 1:296) {
    slice = ((indexes[i] - 1) * 5) + 1
    slice_next = indexes[i] * 5
    a = list()
    a$X = new_data[[i]]$X[slice : slice_next,]
    a$y = new_data[[i]]$y[indexes[i]]
    test[[i]] = a
    b = list()
    b$X = new_data[[i]]$X[-(slice : slice_next),]
    b$y = new_data[[i]]$y[-(indexes[i])]
    train[[i]] = b
}

# Смотрим на тренировочные данные индивида 73, затем на тестовые:
train[[73]]
test[[73]]

# Теперь у нас для каждого индивида имеется тренировочная матрица X размерностью 30 на 11,
# тренировочный вектор y длиною 6, тестовая матрица X размерностью 5 на 11 и тестовый y,
# который мы в будущем можем использовать для проверки качества модели.