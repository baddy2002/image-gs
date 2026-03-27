

## GaussianSplatting2D class


### _init_gaussians method

- self.xy -> La posizione centrale $(x, y)$ di ogni gaussiana.
- self.scale -> La dimensione (lunghezza e larghezza).
- self.rot -> L'angolo di rotazione.
- self.feat -> I "fatures", ovvero il colore (RGB) della gaussiana.

### _log_compression_rate method

- calcola dimensione dell'immaggine non compressa: pixel*num_canali

- calcola dimensione(bit) compressa: num_gaussian*