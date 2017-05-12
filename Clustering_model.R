library(rstan)
library(reshape2)
library(ggplot2) # графики
library(dendextend) # для красивых иерархических деревьев :)
library(extrafont) # шрифты 
font_install("fontcm") # шрифт, который совпадает с латеховским

# Если желаем, можем загрузить все шрифты:
loadfonts()

# Можно на них посмотреть:
fonts()

# Теперь мы хотим кластеризовать индивидов, предпочтения которых описываются 11 коэффициентами бета.
# Попробуем снизить размерность 11-мерного пространства с помощью метода главных компонент:

# Берём беты, полученные ранее в Bayesian_hierarchical_multi_logit.R и выделяем 11 главных компонент:
betas.pca <- prcomp(betas_hier, scale = TRUE) 

# Строим график:

autoplot(betas.pca, loadings = TRUE, loadings.colour = '#40E0D0',
         loadings.label = TRUE, loadings.label.size = 3, xlim = c(-0.15, 0.25))
# На графике явных кластеров нет :( 

# Если мы хотим сохранить график в .pdf с красивым шрифтом, который использует Latex, то пишем следующее:

PCA <- autoplot(betas.pca, loadings = TRUE, loadings.colour = '#40E0D0',
                loadings.label = TRUE, loadings.label.size = 3, xlim = c(-0.15, 0.25) ) + theme(text=element_text(family="CM Roman"))

ggsave("PCA.pdf", PCA, width=4, height=3.5)
embed_fonts("PCA.pdf", outfile="PCA_embed.pdf")  
# Заметим, что с новым шрифтом можно только сохранить график, в R на него никак не посмотреть. 


######################################################


# Теперь попробуем нелинейную технику снижения размерности, а именно, алгоритм t-SNE.
set.seed(78)
tsne_out <- Rtsne(betas_hier, perplexity = 5, dims = 2, theta = 0)

d_tsne = as.data.frame((tsne_out$Y))

# Строим график в осях t-SNE 
ggplot(d_tsne_1, aes(x=V1, y=V2)) +  
    geom_point(size=0.25) + ggtitle("Perplexity = 5") + theme(plot.title = element_text(colour = "#FF0000")) +
    xlab("t-SNE 1") + ylab("t-SNE 2")

# Теперь разделим индивидов на группы с помощью иерархической кластеризации:
d_betas <- dist(betas_hier)
hc_betas <- hclust(d_betas, method = "complete")

# Нарисуем дерево для 3 кластеров
dend <- as.dendrogram(hc_betas)
dend <- color_branches(dend, k=3)
plot(dend)

# Теперь изобразим полученные 3 кластера на графике в осях t-SNE.

# Сохраняем данные под новым именем:
d_tsne_original=d_tsne

# Добавляем в данные информацию о принадлежности к одному из 3 кластеров
d_tsne_original$cl_hierarchical <- factor(cutree(dend, k = 3))

# Создаём функцию для графика
plot_cluster=function(data, var_cluster)  
{
    ggplot(data, aes_string(x="V1", y="V2", color=var_cluster)) +
        geom_point(size=0.25) +
        guides(colour=guide_legend(override.aes=list(size=6))) +
        xlab("t-SNE 1") + ylab("t-SNE 2") +
        ggtitle("") +
        theme(axis.text.x=element_blank(),
              axis.text.y=element_blank(),
              legend.direction = "horizontal", 
              legend.position = "bottom",
              legend.box = "horizontal") + 
        scale_color_manual(values=c("#FF0000", "#696969", "#40E0D0"), name = " ")
}

# Смотрим на получившийся график:
plot_cluster(d_tsne_original, "cl_hierarchical")    


######################################################

# Считаем кластерную модель, где индивиды распределены в соответствии с алгоритмом иерархической кластеризации.

# Берём модель и данные из файла Bayesian_hierarchical_multi_logit.R.

# Напомним, так выглядела модель:
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

# Все данные остаются прежними, кроме L и ll, которые отвечают за кластеризацию:
L_3_clusters <- 3
ll_3_clusters <- cutree(dend, k = 3)

# Запускаем модель:
data_3_clusters <- list(N = N, y = y_train, X = X_train, X_tilde = X_test, D = D, C = C, C_tilde = C_tilde, K = K, L = L_3_clusters, ll = ll_3_clusters)
fit_model_3_clusters<- stan(model_code = bayesian_hierarchical_multi_logit, data = data_3_clusters, iter = 1000, chains = 4) 

# Сохранить вектор бет и предсказаний можно так же, как было сделано в Bayesian_hierarchical_multi_logit.R.