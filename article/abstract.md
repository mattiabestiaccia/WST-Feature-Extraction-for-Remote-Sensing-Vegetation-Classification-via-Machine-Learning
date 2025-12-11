# Multiscale Feature Extraction with Wavelet Scattering Transform for Remote Sensing Vegetation Classification via Machine Learning

## Abstract

Land cover classification via Machine Learning is essential for ecological monitoring, but its reliability is often compromised by noisy imagery and limited datasets, common challenges in UAV acquisitions. Adopting advanced feature extraction is crucial to maintain accuracy under suboptimal conditions. This study evaluates the Wavelet Scattering Transform (WST) as a feature extraction technique for Random Forest-based classification across three coastal sites in the Chesapeake Bay region (Maryland, USA) representing diverse ecological environments.

The experimental framework comprised 1512 classification experiments spanning six noise types, three dataset sizes (5–40 images per class), and three feature extraction methods: advanced statistical descriptors, WST, and a hybrid approach combining both.

Key findings demonstrate that the Hybrid method achieves statistically significant improvements over the statistical baseline. The Hybrid approach exhibits superior noise robustness, degrading slower than Advanced Statistics. Under extreme data scarcity, all methods retain strong data efficiency.

However, WST alone does not significantly outperform traditional statistical features, and its substantial computational overhead is not justified in low-noise conditions. These results provide practical guidance for UAV-based monitoring: Hybrid features are recommended for high-noise scenarios with GPU acceleration, while simpler statistical approaches suffice for clean imagery or resource-constrained deployments.

**Keywords**: Feature extraction, Wavelet Scattering Transform, Vegetation Classification, Ecological Restoration, Wetland Monitoring, UAV Remote Sensing, Machine Learning, Drones
