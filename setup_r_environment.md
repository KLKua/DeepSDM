# Install All the R Packages through Bash

## 1. Install the Latest R
Follow the instructions on the [R Project website](https://cran.csie.ntu.edu.tw/).

### Steps
1. Update indices:
   ```bash
   sudo apt update -qq
   ```

2. Install two helper packages:
   ```bash
   sudo apt install --no-install-recommends software-properties-common dirmngr
   ```

3. Add the signing key (by Michael Rutter) for these repositories:
   To verify the key, run:
   ```bash
   gpg --show-keys /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
   ```
   The fingerprint should be:
   ```
   E298A3A825C0D65DFD57CBB651716619E084DAB9
   ```
   Add the key:
   ```bash
   wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
   ```

4. Add the R 4.0 repository from CRAN (adjust `focal` to `groovy` or `bionic` if needed):
   ```bash
   sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
   ```

5. Install R:
   ```bash
   sudo apt install --no-install-recommends r-base
   ```

## 2. Install `raster`
1. Install `gdal` first for the necessary package `terra`:
   ```bash
   sudo apt update
   sudo apt install gdal-bin libgdal-dev
   ```

2. Install the `raster` package:
   ```bash
   sudo Rscript -e 'install.packages("raster")'
   ```

## 3. Install `dismo`
```bash
sudo Rscript -e 'install.packages("dismo")'
```

## 4. Install `pROC`
```bash
sudo Rscript -e 'install.packages("pROC")'
```

## 5. Install `tidyverse`
1. Install the dependencies for the required package `ragg`:
   ```bash
   sudo apt update
   sudo apt install libharfbuzz-dev libfribidi-dev libfontconfig1-dev
   ```

2. Install the `tidyverse` package:
   ```bash
   sudo Rscript -e 'install.packages("tidyverse")'
   ```

## 6. Install `rjson`
```bash
sudo Rscript -e 'install.packages("rjson")'
```

## 7. Install `yaml`
```bash
sudo Rscript -e 'install.packages("yaml")'
```

## 8. Install `rJava`
1. Install Java:
   ```bash
   sudo apt update
   sudo apt install default-jdk
   ```

2. Check the Java path:
   ```bash
   which java
   ```

3. Find the Java path and set it:
   ```bash
   export JAVA_HOME=/usr/lib/jvm/default-java
   export PATH=$JAVA_HOME/bin:$PATH
   ```

4. Add the path to your `.bashrc` file:
   ```bash
   echo 'export JAVA_HOME=/usr/lib/jvm/default-java' >> ~/.bashrc
   echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

5. Check the configuration:
   ```bash
   R CMD javareconf
   ```

6. Install the `rJava` package:
   ```bash
   sudo Rscript -e 'install.packages("rJava")'
   ```

## 9. Install `hdf5r`
```bash
sudo Rscript -e 'install.packages("hdf5r")'
```

## 10. Install `arrow`
```bash
sudo Rscript -e 'install.packages("arrow")'
```