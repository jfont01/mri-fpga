/******************************************************************************
 * KV260 AXI DMA - medicion de ancho de banda
 *
 *   DDR -> MM2S -> AXI4-Stream Data FIFO -> S2MM -> DDR
 *
 * Barre tamanos de transferencia y reporta:
 *   - tiempo de DMA
 *   - ancho de banda de pipeline  (N / t)
 *   - trafico total sobre la DDR  (2N / t, porque lee N y escribe N)
 *   - tiempo de mantenimiento de cache, medido aparte
 *
 * Bare-metal, polling, sin interrupciones, sin Scatter/Gather.
 ******************************************************************************/

#include "xparameters.h"
#include "xaxidma.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xstatus.h"
#include "xtime_l.h"

/* ------------------------------------------------------------------------- */
/* Configuracion                                                              */
/* ------------------------------------------------------------------------- */

#define MAX_TRANSFER_BYTES      (8U * 1024U * 1024U)   /* 8 MiB */
#define BUFFER_ALIGNMENT        64U                    /* linea de cache A53 */

#define WARMUP_RUNS             1U
#define MEASURED_RUNS           5U

/* El diseno mapea solo DDR_LOW en los espacios de direcciones del DMA. */
#define DMA_DDR_LOW_LAST_ADDR   ((UINTPTR)0x7FFFFFFFUL)

/* Tamanos del barrido. Todos multiplos de 16 B (ancho MM del DMA). */
static const u32 SweepSizes[] = {
    4U    * 1024U,
    16U   * 1024U,
    64U   * 1024U,
    256U  * 1024U,
    1024U * 1024U,
    2048U * 1024U,
    4096U * 1024U,
    8192U * 1024U
};
#define NUM_SWEEP_SIZES  (sizeof(SweepSizes) / sizeof(SweepSizes[0]))

/* ------------------------------------------------------------------------- */
/* Estado global                                                              */
/* ------------------------------------------------------------------------- */

static XAxiDma AxiDma;

static u8 TxBuffer[MAX_TRANSFER_BYTES]
    __attribute__((aligned(BUFFER_ALIGNMENT)));
static u8 RxBuffer[MAX_TRANSFER_BYTES]
    __attribute__((aligned(BUFFER_ALIGNMENT)));

/* ------------------------------------------------------------------------- */
/* Verificacion de ubicacion de buffers                                       */
/* ------------------------------------------------------------------------- */

static int RangeIsInDdrLow(UINTPTR Address, u32 Length)
{
    UINTPTR LastAddress;

    if (Length == 0U) {
        return 0;
    }
    LastAddress = Address + (UINTPTR)Length - 1U;

    if (Address > DMA_DDR_LOW_LAST_ADDR) {
        return 0;
    }
    if (LastAddress > DMA_DDR_LOW_LAST_ADDR) {
        return 0;
    }
    return 1;
}

static int CheckBufferPlacement(XAxiDma_Config *CfgPtr)
{
    UINTPTR TxAddr = (UINTPTR)TxBuffer;
    UINTPTR RxAddr = (UINTPTR)RxBuffer;
    u32 Mm2sAlignment = (u32)CfgPtr->Mm2SDataWidth / 8U;
    u32 S2mmAlignment = (u32)CfgPtr->S2MmDataWidth / 8U;

    xil_printf("\r\nBuffers:\r\n");
    xil_printf("  TX = 0x%x\r\n", (u32)TxAddr);
    xil_printf("  RX = 0x%x\r\n", (u32)RxAddr);
    xil_printf("  tamano maximo = %d KiB c/u\r\n",
               MAX_TRANSFER_BYTES / 1024U);

    if (!RangeIsInDdrLow(TxAddr, MAX_TRANSFER_BYTES)) {
        xil_printf("ERROR: TX fuera del espacio DDR_LOW del DMA\r\n");
        return XST_FAILURE;
    }
    if (!RangeIsInDdrLow(RxAddr, MAX_TRANSFER_BYTES)) {
        xil_printf("ERROR: RX fuera del espacio DDR_LOW del DMA\r\n");
        return XST_FAILURE;
    }

    if (!CfgPtr->HasMm2SDRE) {
        if ((Mm2sAlignment == 0U) ||
            ((TxAddr % (UINTPTR)Mm2sAlignment) != 0U)) {
            xil_printf("ERROR: TX sin alinear para MM2S\r\n");
            return XST_FAILURE;
        }
    }
    if (!CfgPtr->HasS2MmDRE) {
        if ((S2mmAlignment == 0U) ||
            ((RxAddr % (UINTPTR)S2mmAlignment) != 0U)) {
            xil_printf("ERROR: RX sin alinear para S2MM\r\n");
            return XST_FAILURE;
        }
    }

    xil_printf("Ubicacion y alineacion OK\r\n");
    return XST_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* Diagnostico                                                                */
/* ------------------------------------------------------------------------- */

static void DumpDmaStatus(XAxiDma *DmaPtr)
{
    xil_printf("MM2S_DMASR = 0x%x\r\n",
        XAxiDma_ReadReg(DmaPtr->RegBase + XAXIDMA_TX_OFFSET,
                        XAXIDMA_SR_OFFSET));
    xil_printf("S2MM_DMASR = 0x%x\r\n",
        XAxiDma_ReadReg(DmaPtr->RegBase + XAXIDMA_RX_OFFSET,
                        XAXIDMA_SR_OFFSET));
}

/* ------------------------------------------------------------------------- */
/* Inicializacion                                                             */
/* ------------------------------------------------------------------------- */

static int InitDma(void)
{
    XAxiDma_Config *CfgPtr;
    int Status;

#ifndef SDT
    CfgPtr = XAxiDma_LookupConfig(XPAR_AXIDMA_0_DEVICE_ID);
#else
    CfgPtr = XAxiDma_LookupConfig(XPAR_XAXIDMA_0_BASEADDR);
#endif
    if (CfgPtr == NULL) {
        xil_printf("ERROR: no se encontro la configuracion del AXI DMA\r\n");
        return XST_FAILURE;
    }

    Status = XAxiDma_CfgInitialize(&AxiDma, CfgPtr);
    if (Status != XST_SUCCESS) {
        xil_printf("ERROR: XAxiDma_CfgInitialize fallo: %d\r\n", Status);
        return XST_FAILURE;
    }

    xil_printf("\r\nAXI DMA:\r\n");
    xil_printf("  MM2S / S2MM    : %d / %d\r\n",
               CfgPtr->HasMm2S, CfgPtr->HasS2Mm);
    xil_printf("  Scatter/Gather : %d\r\n", CfgPtr->HasSg);
    xil_printf("  ancho MM2S     : %d bits\r\n", CfgPtr->Mm2SDataWidth);
    xil_printf("  ancho S2MM     : %d bits\r\n", CfgPtr->S2MmDataWidth);
    xil_printf("  DRE MM2S/S2MM  : %d / %d\r\n",
               CfgPtr->HasMm2SDRE, CfgPtr->HasS2MmDRE);

    if (!CfgPtr->HasMm2S || !CfgPtr->HasS2Mm) {
        xil_printf("ERROR: falta un canal del DMA\r\n");
        return XST_FAILURE;
    }
    if (XAxiDma_HasSg(&AxiDma)) {
        xil_printf("ERROR: el DMA esta en modo Scatter/Gather\r\n");
        return XST_FAILURE;
    }

    XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);
    XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);

    return CheckBufferPlacement(CfgPtr);
}

/* ------------------------------------------------------------------------- */
/* Cronometro                                                                 */
/* ------------------------------------------------------------------------- */
/*
 * XTime_GetTime() lee el contador generico del Cortex-A53 (CNTPCT_EL0).
 * COUNTS_PER_SECOND lo define la BSP; en esta placa es ~100 MHz,
 * o sea ~10 ns de resolucion. Sin punto flotante: xil_printf no lo imprime.
 */

static u32 TicksToMicros(XTime Ticks)
{
    return (u32)(((u64)Ticks * 1000000ULL) / (u64)COUNTS_PER_SECOND);
}

static u32 ComputeMBps(u32 Bytes, XTime Ticks)
{
    if (Ticks == 0U) {
        return 0U;
    }
    /* bytes/s = Bytes * COUNTS_PER_SECOND / Ticks, luego a MB/s */
    return (u32)((((u64)Bytes * (u64)COUNTS_PER_SECOND) / (u64)Ticks)
                 / 1000000ULL);
}

/* ------------------------------------------------------------------------- */
/* Una transferencia cronometrada                                             */
/* ------------------------------------------------------------------------- */

static int TimedTransfer(u32 Bytes, XTime *DmaTicks, XTime *FlushTicks)
{
    XTime t0, t1, t2, t3;
    int   Status;

    /* --- mantenimiento de cache, medido aparte --- */
    XTime_GetTime(&t0);
    Xil_DCacheFlushRange((UINTPTR)TxBuffer, Bytes);
    Xil_DCacheFlushRange((UINTPTR)RxBuffer, Bytes);
    XTime_GetTime(&t1);

    /* --- transferencia --- */
    XTime_GetTime(&t2);

    /* S2MM primero: arma el destino antes de que MM2S produzca datos. */
    Status = XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)RxBuffer, Bytes,
                                    XAXIDMA_DEVICE_TO_DMA);
    if (Status != XST_SUCCESS) {
        xil_printf("ERROR: no arranco S2MM: %d\r\n", Status);
        return XST_FAILURE;
    }

    Status = XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)TxBuffer, Bytes,
                                    XAXIDMA_DMA_TO_DEVICE);
    if (Status != XST_SUCCESS) {
        xil_printf("ERROR: no arranco MM2S: %d\r\n", Status);
        return XST_FAILURE;
    }

    /*
     * Espera activa. No se usa usleep(): su sobrecarga es mucho mayor
     * que 1 us y contaminaria la medicion.
     */
    while (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE) ||
           XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA)) {
        u32 TxSr = XAxiDma_ReadReg(AxiDma.RegBase + XAXIDMA_TX_OFFSET,
                                   XAXIDMA_SR_OFFSET);
        u32 RxSr = XAxiDma_ReadReg(AxiDma.RegBase + XAXIDMA_RX_OFFSET,
                                   XAXIDMA_SR_OFFSET);
        if (((TxSr | RxSr) & XAXIDMA_ERR_ALL_MASK) != 0U) {
            xil_printf("ERROR de DMA: MM2S=0x%x  S2MM=0x%x\r\n", TxSr, RxSr);
            return XST_FAILURE;
        }
    }

    XTime_GetTime(&t3);

    *FlushTicks = t1 - t0;
    *DmaTicks   = t3 - t2;

    return XST_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* Barrido                                                                    */
/* ------------------------------------------------------------------------- */

static int RunSweep(void)
{
    u32 i, j, run;

    xil_printf("\r\n");
    xil_printf("=========================================================="
               "==========\r\n");
    xil_printf(" BARRIDO DE ANCHO DE BANDA   (loopback DDR -> PL -> DDR)\r\n");
    xil_printf("=========================================================="
               "==========\r\n\r\n");
    xil_printf("    tamano    t_dma      pipeline    trafico DDR   "
               "t_flush\r\n");
    xil_printf("     (KiB)     (us)        (MB/s)       (MB/s)      "
               "(us)\r\n");
    xil_printf("   --------  --------    --------    -----------   "
               "--------\r\n");

    for (i = 0U; i < NUM_SWEEP_SIZES; i++) {
        u32   Bytes    = SweepSizes[i];
        XTime BestDma  = 0;
        XTime FlushAcc = 0;

        /* patron determinista */
        for (j = 0U; j < Bytes; j++) {
            TxBuffer[j] = (u8)(((j * 13U) + 0x5AU) & 0xFFU);
            RxBuffer[j] = 0xA5U;
        }

        /* calentamiento: se descarta */
        for (run = 0U; run < WARMUP_RUNS; run++) {
            XTime d, f;
            if (TimedTransfer(Bytes, &d, &f) != XST_SUCCESS) {
                DumpDmaStatus(&AxiDma);
                return XST_FAILURE;
            }
        }

        /* corridas medidas: se guarda el mejor tiempo de DMA */
        for (run = 0U; run < MEASURED_RUNS; run++) {
            XTime d, f;
            if (TimedTransfer(Bytes, &d, &f) != XST_SUCCESS) {
                DumpDmaStatus(&AxiDma);
                return XST_FAILURE;
            }
            if ((BestDma == 0) || (d < BestDma)) {
                BestDma = d;
            }
            FlushAcc += f;
        }

        /* verificacion de datos */
        Xil_DCacheInvalidateRange((UINTPTR)RxBuffer, Bytes);
        for (j = 0U; j < Bytes; j++) {
            if (RxBuffer[j] != TxBuffer[j]) {
                xil_printf("\r\nERROR DE DATOS: tamano %d, indice %d, "
                           "esperado 0x%x, recibido 0x%x\r\n",
                           Bytes, j, (u32)TxBuffer[j], (u32)RxBuffer[j]);
                return XST_FAILURE;
            }
        }

        xil_printf("   %8d  %8d    %8d    %11d   %8d\r\n",
                   Bytes / 1024U,
                   TicksToMicros(BestDma),
                   ComputeMBps(Bytes, BestDma),
                   ComputeMBps(Bytes * 2U, BestDma),
                   TicksToMicros(FlushAcc / MEASURED_RUNS));
    }

    xil_printf("\r\n");
    return XST_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* main                                                                       */
/* ------------------------------------------------------------------------- */

int main(void)
{
    int Status;

    xil_printf("\r\n");
    xil_printf("========================================\r\n");
    xil_printf(" KV260 AXI DMA - ANCHO DE BANDA\r\n");
    xil_printf("========================================\r\n");

    Status = InitDma();
    if (Status != XST_SUCCESS) {
        xil_printf("\r\nFALLO LA INICIALIZACION\r\n");
        return XST_FAILURE;
    }

    xil_printf("\r\nCOUNTS_PER_SECOND = %d Hz\r\n", (u32)COUNTS_PER_SECOND);

    Status = RunSweep();
    if (Status != XST_SUCCESS) {
        xil_printf("\r\nEL BARRIDO FALLO\r\n");
        DumpDmaStatus(&AxiDma);
        return XST_FAILURE;
    }

    xil_printf("========================================\r\n");
    xil_printf(" BARRIDO COMPLETO - datos verificados OK\r\n");
    xil_printf("========================================\r\n");

    return XST_SUCCESS;
}