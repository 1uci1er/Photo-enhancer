┌─────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE(S)                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              IMAGE ANALYZER MODULE                          │
│  • Resolution Detection    • Compression Artifacts         │
│  • Noise Level Analysis    • Face Detection               │
│  • Blur Assessment        • Scene Type Classification      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│             PREPROCESSING PIPELINE                          │
│  • EXIF Cleanup           • Auto-Orientation              │
│  • Color Space Conversion • Gamma Correction              │
│  • Dynamic Range Analysis • Noise Pre-filtering           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│            DYNAMIC MODEL ORCHESTRATOR                       │
│  Routes to appropriate enhancement models based on analysis │
└─────────┬─────────────────────────────────────┬─────────────┘
          │                                     │
┌─────────▼──────────┐                 ┌───────▼──────────┐
│   FACE PIPELINE    │                 │ BACKGROUND       │
│ • GFPGAN/CodeFormer│                 │ PIPELINE         │
│ • Face Super-Res   │                 │ • Real-ESRGAN    │
│ • Detail Recovery  │                 │ • SwinIR         │
│ • Skin Enhancement │                 │ • HAT/SRFormer   │
└─────────┬──────────┘                 └───────┬──────────┘
          │                                     │
┌─────────▼───────────────────────────────────▼─────────────┐
│              POST-PROCESSING FUSION                       │
│  • Seamless Face-Background Blending                     │
│  • Color Harmony Adjustment                              │
│  • Final Quality Enhancement                             │
└─────────────────────┬─────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                OUTPUT IMAGE(S)                              │
│         Enhanced to Target Resolution                       │
└─────────────────────────────────────────────────────────────┘
