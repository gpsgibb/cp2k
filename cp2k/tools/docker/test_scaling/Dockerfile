FROM cp2k/toolchain:latest

# author: Ole Schuett

# download and compile cp2k snapshot
WORKDIR /opt/
RUN wget -O cp2k-master.zip https://github.com/cp2k/cp2k/archive/master.zip && \
    unzip cp2k-master.zip  && \
    rm cp2k-master.zip

WORKDIR /opt/cp2k-master/cp2k/arch
RUN ln -vs /opt/cp2k-toolchain/install/arch/local* .

WORKDIR /opt/cp2k-master/cp2k/makefiles
RUN source /opt/cp2k-toolchain/install/setup  && \
    make -j VERSION="popt psmp ssmp" cp2k     && \
    rm -rf ../lib/ ../exe/

# install Debian packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-numpy                                                \
    rsync                                                        \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/cp2k_test_scaling
COPY cmd.sh .
CMD ["/opt/cp2k_test_scaling/cmd.sh"]

#EOF
