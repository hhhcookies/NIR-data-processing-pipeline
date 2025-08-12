import config as cfg
import NIR_processing as nir
import matplotlib.pyplot as plt

def main():
    # 1. generate metadata and spectrum data
    print('Loading data.....')
    metadata, data_nir = nir.data_import(cfg.DATA_FILE)

    # 2. Summarize metadata
    print('Summarizing metadata...')
    metadata_info = nir.metadata_summary(cfg.FORMULATION, metadata)
    

    # 3.Plot raw spectrum
    print("Plotting raw spectra...")
    nir.plot_spectrum(metadata, data_nir, "Raw Spectrum")

    # 4.Pre-process spectrum
    print("Preprocessing spectra...")
    processed = nir.preprocessing_pipeline(metadata,data_nir)

    # 5.color-coded spectrum
    print("Plotting spectra by composition...")
    nir.plot_spectrum(metadata, processed, "per Arginine", colorby="perc_A")
    # nir.plot_spectrum(metadata, processed, "per Sucrose", colorby="perc_S")

    # 6.PCA, tune number of components
    nir.pca_hyper(processed)

    # 7.Outlier detection
    print("Detecting outliers...")
    dist_df, score, loading = nir.Q_T2_outlier_detect(processed,n_components=3,CI_percentile=95,outlier_threshold=99)

    # 8.PCA score plots:
    print("Plotting PCA score plots...")
    nir.sample_plot(metadata, score, "Type", True)
    nir.sample_plot(metadata, score, "Operator", True)
    nir.sample_plot(metadata, score, "perc_A", False)
    nir.sample_plot(metadata, score, "perc_S", False)

    # 9.find optimal number of components for PLS
    print("Optimizing PLS components...")
    cv_r2_percA = nir.optimum_variables(processed.values, metadata["perc_A"].values, max_components=20)
    # cv_r2_percS = nir.optimum_variables(processed.values, metadata["perc_S"].values, max_components=20)

    # 10.Train PLS models
    print("Training PLS models...")
    pls = nir.PLS(n_components=2)
    pls.fit(processed.values, metadata["perc_A"])

    # pls_S = nir.PLS(n_components=5)
    # pls_S.fit(processed.values, metadata["perc_S"])

    # 11.Prediction and residual plot
    print("Predicting for perc_A...")
    nir.prediction(pls, metadata, processed, "perc_A", lb=-0.5, ub=3)

    # print("Predicting for perc_S...")
    # nir.prediction(pls_S, metadata, processed, "perc_S", lb=-0.5, ub=9)

    print("Pipeline completed successfully!")
    plt.show()

if __name__ == '__main__':
    main()













