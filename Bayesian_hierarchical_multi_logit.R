library("rstan") # врубим на полную мощь Байеса!
library("shinystan") # и на полную громкость интерактивный анализ полученной цепи
library("tibble")
library("dplyr")
library("HSAUR")
library("psych")

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Строим модель языком Stan`а. Она состоит из 4х блоков: данные, параметры, 
# модель (внутри которой задаются априорные распределения и сам мультиномиальный логит),
# блок для предсказаний на тестовой выборке

bayesian_hierarchical_multi_logit <- "
data {
int N;                           // количество индивидов
int<lower=2> K;                  // количество альтернатив
int<lower=1> D;                  // количество атрибутов + 1 (свободный коэффициент) 
int<lower=1> C;                  // количество карточек для тренировочной выборки
int<lower=1> C_tilde;            // количество карточек для тестовой выборки
int<lower=1> L;                  // количество групп 
int<lower=1,upper=L> ll[N];      // принадлежность каждого индивида к группе
matrix[K,D] X[N, C];             // матрица X из тренировочной выборки
matrix[K,D] X_tilde[N, C_tilde]; // матрица X из тестовой выборки
int<lower=1,upper=5> y[N, C];    // выбор индивида для каждой карточки
}
parameters {
vector[D] mu;                 // вектор средних
corr_matrix[D] Omega;         // корреляционная матрица
vector<lower=0>[D] tau;       // вектор стандартных отклонений
vector[D] beta[L];            // индивидуальные коэффициенты
}
model {
tau ~ cauchy(0, 2.5);
Omega ~ lkj_corr(2);
mu ~ normal(0, 5);
for (l in 1:L)
beta[l] ~ multi_normal(mu, quad_form_diag(Omega, tau));
for (n in 1:N)
for (c in 1:C)
y[n, c] ~ categorical_logit(X[n, c]*beta[ll[n]]);
}
generated quantities {
int<lower=1,upper=5> y_tilde[N, C_tilde];
for (n in 1:N)
for (c in 1:C_tilde)
y_tilde[n, c] = categorical_rng(softmax(X_tilde[n, c]*beta[ll[n]]));
}
"

# Теперь нам нужно задать все параметры, описанные в блоке Data:
N <- 296
K <- 5
D <- 11
C <- 6
C_tilde <- 1
L <- 296
ll <- c(1:296)

# Нужно задать тренировочную матрицу X так, как хочет Stan.
X_train <- rep(0, 296 * 6 * 5 * 11)
dim(X_train) <- c(296, 6, 5, 11)

for (n in 1:296) {
    for (c in 1:6) {
        for (k in 1:5) {
            for (d in 1:11) {
                X_train[n,c,k,d] <- train[[n]]$X[(c - 1) * 5 + k, d] #train был взят из файла Data_preparation.R
            }
        }
    }
}


# Теперь зададим тестовую матрицу X так же:
X_test <- rep(0, 296 * 1 * 5 * 11)
dim(X_test) <- c(296, 1, 5, 11)

for (n in 1:296) {
    for (c in 1:1) {
        for (k in 1:5) {
            for (d in 1:11) {
                X_test[n,c,k,d] <- test[[n]]$X[(c - 1) * 5 + k, d]
            }
        }
    }
}
X_test

# Наконец, разберёмся с вектором y.
y_train <- rep(0, 296 * 6)
dim(y_train) <- c(296, 6)

for (n in 1:296) {
    for (c in 1:6) {
        y_train[n,c] <- train[[n]]$y[c]
    }
}
y_train

# Теперь можем запускать модель и примерять роль Хатико...
data_for_model <- list(N = N, y = y_train, X = X_train, X_tilde = X_test, D = D, C = C, C_tilde = C_tilde, K = K, L = L, ll = ll)
fit_model <- stan(model_code = bayesian_hierarchical_multi_logit, data = data_for_hierarchical_logit, iter = 1000, chains = 4) 

# Если хватило терпения, можно извлекать результаты.

list_result_hier_logit <- rstan::extract(fit_model)
as.data.frame(list_result_hier_logit)

# Усредняем беты по 2000 итераций:
betas_hier <- rep(0, 296 * 11)
dim(betas_hier) <- c(296, 11)

for (j in 1:296) {
    for (k in 1:11) {
        betas_hier[j, k] <- mean(list_result_hier_logit$beta[,j,k])
    }
}

# Сохраняем беты.
colnames(betas_hier) <- colnames(new_data[[1]]$X)
write.csv(betas_hier, file = "betas_hier")


# Для каждого индивида считаем вероятности выбрать ту или иную альтернативу на карточках из тестовой выборки
r <- array(dim = c(2000, 296))
p <- array(dim = c(296, 5))
for (n in 1:296){
    for (i in 1:2000){
        for (a in 1:5) {
            r[i, n] <- list_result_hier_logit$y_tilde[i, n, 1]
            p[n, a] <- sum(r[,n][r[,n] == a ])/(a*2000)
        }
    }
}

# Посмотрим на получившиеся вероятности:
p

# Сохраним их.
colnames(p) <- c(1, 2, 3, 4, 5)
col <- colnames(p)
write.csv(p, file = "probability_of_choice") 
