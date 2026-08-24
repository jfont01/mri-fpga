/******************************************************************************
 * KV260 AXI DMA V0
 *
 * Test:
 *
 * DDR -> MM2S -> AXI4-Stream Data FIFO -> S2MM -> DDR
 *
 * - Bare-metal
 * - Simple DMA (Scatter/Gather disabled)
 * - Polling
 * - No interrupts
 * - 4096-byte transfer
 * - Cache maintenance
 *
 ******************************************************************************/

#include "xparameters.h"
#include "xaxidma.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xstatus.h"
#include "sleep.h"

/* ------------------------------------------------------------------------- */
/* Test configuration                                                        */
/* ------------------------------------------------------------------------- */

#define TRANSFER_BYTES          4096U

/*
 * A53 cache line = 64 bytes.
 *
 * This also comfortably satisfies the DMA's 128-bit = 16-byte alignment
 * requirement when DRE is disabled.
 */
#define BUFFER_ALIGNMENT        64U

/*
 * Our Vivado design maps only DDR_LOW into the DMA address spaces:
 *
 *     0x00000000 - 0x7FFFFFFF
 *
 * DDR_HIGH, OCM and QSPI are excluded.
 */
#define DMA_DDR_LOW_LAST_ADDR   ((UINTPTR)0x7FFFFFFFUL)

/*
 * Timeout for polling.
 *
 * We sleep 1 us per iteration, so this is approximately 1 second.
 */
#define POLL_TIMEOUT_US         1000000U


/* ------------------------------------------------------------------------- */
/* DMA instance                                                              */
/* ------------------------------------------------------------------------- */

static XAxiDma AxiDma;


/* ------------------------------------------------------------------------- */
/* DMA buffers                                                               */
/* ------------------------------------------------------------------------- */

/*
 * Global/static buffers avoid stack allocation.
 *
 * 64-byte alignment:
 *   - good for Cortex-A53 cache maintenance
 *   - also satisfies our 16-byte AXI DMA alignment requirement
 */
static u8 TxBuffer[TRANSFER_BYTES]
    __attribute__((aligned(BUFFER_ALIGNMENT)));

static u8 RxBuffer[TRANSFER_BYTES]
    __attribute__((aligned(BUFFER_ALIGNMENT)));


/* ------------------------------------------------------------------------- */
/* Helper functions                                                          */
/* ------------------------------------------------------------------------- */

static void FillBuffers(void)
{
    u32 i;

    for (i = 0U; i < TRANSFER_BYTES; i++) {

        /*
         * Deterministic pattern.
         *
         * Not simply all zeros, so failures are easier to identify.
         */
        TxBuffer[i] = (u8)(((i * 13U) + 0x5AU) & 0xFFU);

        /*
         * Fill destination with a recognizable value before DMA.
         */
        RxBuffer[i] = 0xA5U;
    }
}


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

    u32 Mm2sAlignment;
    u32 S2mmAlignment;

    /*
     * Memory-mapped widths are reported in bits.
     * Convert them to bytes.
     */
    Mm2sAlignment = (u32)CfgPtr->Mm2SDataWidth / 8U;
    S2mmAlignment = (u32)CfgPtr->S2MmDataWidth / 8U;

    xil_printf("\r\nBuffer information:\r\n");
    xil_printf("  TX address = 0x%x\r\n", (u32)TxAddr);
    xil_printf("  RX address = 0x%x\r\n", (u32)RxAddr);
    xil_printf("  Length     = %d bytes\r\n", TRANSFER_BYTES);

    /*
     * Check that linker placed both buffers in the DDR region
     * visible from the DMA.
     */
    if (!RangeIsInDdrLow(TxAddr, TRANSFER_BYTES)) {
        xil_printf("ERROR: TX buffer is outside DMA DDR_LOW address space\r\n");
        return XST_FAILURE;
    }

    if (!RangeIsInDdrLow(RxAddr, TRANSFER_BYTES)) {
        xil_printf("ERROR: RX buffer is outside DMA DDR_LOW address space\r\n");
        return XST_FAILURE;
    }

    /*
     * If DRE is disabled, DMA addresses must be aligned to
     * memory-map data width.
     */
    if (!CfgPtr->HasMm2SDRE) {

        if ((Mm2sAlignment == 0U) ||
            ((TxAddr % (UINTPTR)Mm2sAlignment) != 0U)) {

            xil_printf("ERROR: TX buffer is not MM2S aligned\r\n");
            return XST_FAILURE;
        }
    }

    if (!CfgPtr->HasS2MmDRE) {

        if ((S2mmAlignment == 0U) ||
            ((RxAddr % (UINTPTR)S2mmAlignment) != 0U)) {

            xil_printf("ERROR: RX buffer is not S2MM aligned\r\n");
            return XST_FAILURE;
        }
    }

    xil_printf("Buffer placement/alignment OK\r\n");

    return XST_SUCCESS;
}


static void DumpDmaStatus(XAxiDma *DmaPtr)
{
    u32 Mm2sStatus;
    u32 S2mmStatus;

    Mm2sStatus =
        XAxiDma_ReadReg(DmaPtr->RegBase + XAXIDMA_TX_OFFSET,
                        XAXIDMA_SR_OFFSET);

    S2mmStatus =
        XAxiDma_ReadReg(DmaPtr->RegBase + XAXIDMA_RX_OFFSET,
                        XAXIDMA_SR_OFFSET);

    xil_printf("MM2S_DMASR = 0x%x\r\n", Mm2sStatus);
    xil_printf("S2MM_DMASR = 0x%x\r\n", S2mmStatus);
}


static int WaitForDma(XAxiDma *DmaPtr)
{
    u32 Timeout = POLL_TIMEOUT_US;

    while (Timeout > 0U) {

        u32 Mm2sStatus;
        u32 S2mmStatus;

        Mm2sStatus =
            XAxiDma_ReadReg(DmaPtr->RegBase + XAXIDMA_TX_OFFSET,
                            XAXIDMA_SR_OFFSET);

        S2mmStatus =
            XAxiDma_ReadReg(DmaPtr->RegBase + XAXIDMA_RX_OFFSET,
                            XAXIDMA_SR_OFFSET);

        /*
         * Detect DMA errors instead of waiting forever.
         */
        if ((Mm2sStatus & XAXIDMA_ERR_ALL_MASK) != 0U) {

            xil_printf("ERROR: MM2S DMA error\r\n");
            DumpDmaStatus(DmaPtr);

            return XST_FAILURE;
        }

        if ((S2mmStatus & XAXIDMA_ERR_ALL_MASK) != 0U) {

            xil_printf("ERROR: S2MM DMA error\r\n");
            DumpDmaStatus(DmaPtr);

            return XST_FAILURE;
        }

        /*
         * Transfer is done when neither channel is busy.
         */
        if ((!XAxiDma_Busy(DmaPtr, XAXIDMA_DMA_TO_DEVICE)) &&
            (!XAxiDma_Busy(DmaPtr, XAXIDMA_DEVICE_TO_DMA))) {

            return XST_SUCCESS;
        }

        usleep(1U);

        Timeout--;
    }

    xil_printf("ERROR: DMA timeout\r\n");

    DumpDmaStatus(DmaPtr);

    return XST_FAILURE;
}


static int CheckData(void)
{
    u32 i;

    for (i = 0U; i < TRANSFER_BYTES; i++) {

        if (RxBuffer[i] != TxBuffer[i]) {

            xil_printf("\r\nDATA MISMATCH\r\n");
            xil_printf("Index    = %d\r\n", i);
            xil_printf("Expected = 0x%x\r\n", (u32)TxBuffer[i]);
            xil_printf("Received = 0x%x\r\n", (u32)RxBuffer[i]);

            return XST_FAILURE;
        }
    }

    return XST_SUCCESS;
}


/* ------------------------------------------------------------------------- */
/* DMA initialization                                                        */
/* ------------------------------------------------------------------------- */

static int InitDma(void)
{
    XAxiDma_Config *CfgPtr;
    int Status;

    /*
     * AMD currently supports two standalone configuration flows:
     *
     * Classic/XSCT:
     *      lookup by DEVICE_ID
     *
     * System Device Tree (SDT):
     *      lookup by BASEADDR
     *
     * Support both so the program works with either generated platform.
     */
#ifndef SDT

    CfgPtr = XAxiDma_LookupConfig(XPAR_AXIDMA_0_DEVICE_ID);

#else

    CfgPtr = XAxiDma_LookupConfig(XPAR_XAXIDMA_0_BASEADDR);

#endif

    if (CfgPtr == NULL) {

        xil_printf("ERROR: AXI DMA configuration not found\r\n");

        return XST_FAILURE;
    }

    /*
     * Initializes driver and resets hardware.
     */
    Status = XAxiDma_CfgInitialize(&AxiDma, CfgPtr);

    if (Status != XST_SUCCESS) {

        xil_printf("ERROR: XAxiDma_CfgInitialize failed: %d\r\n",
                   Status);

        return XST_FAILURE;
    }

    xil_printf("\r\nAXI DMA configuration:\r\n");

    xil_printf("  MM2S present   : %d\r\n", CfgPtr->HasMm2S);
    xil_printf("  S2MM present   : %d\r\n", CfgPtr->HasS2Mm);
    xil_printf("  Scatter/Gather : %d\r\n", CfgPtr->HasSg);

    xil_printf("  MM2S width     : %d bits\r\n",
               CfgPtr->Mm2SDataWidth);

    xil_printf("  S2MM width     : %d bits\r\n",
               CfgPtr->S2MmDataWidth);

    xil_printf("  MM2S DRE       : %d\r\n",
               CfgPtr->HasMm2SDRE);

    xil_printf("  S2MM DRE       : %d\r\n",
               CfgPtr->HasS2MmDRE);

    /*
     * Hardware sanity checks.
     */
    if (!CfgPtr->HasMm2S) {

        xil_printf("ERROR: MM2S channel not present\r\n");

        return XST_FAILURE;
    }

    if (!CfgPtr->HasS2Mm) {

        xil_printf("ERROR: S2MM channel not present\r\n");

        return XST_FAILURE;
    }

    if (XAxiDma_HasSg(&AxiDma)) {

        xil_printf("ERROR: DMA is configured for Scatter/Gather\r\n");

        return XST_FAILURE;
    }

    /*
     * V0 uses polling only.
     */
    XAxiDma_IntrDisable(&AxiDma,
                        XAXIDMA_IRQ_ALL_MASK,
                        XAXIDMA_DMA_TO_DEVICE);

    XAxiDma_IntrDisable(&AxiDma,
                        XAXIDMA_IRQ_ALL_MASK,
                        XAXIDMA_DEVICE_TO_DMA);

    Status = CheckBufferPlacement(CfgPtr);

    if (Status != XST_SUCCESS) {
        return XST_FAILURE;
    }

    return XST_SUCCESS;
}


/* ------------------------------------------------------------------------- */
/* Main                                                                      */
/* ------------------------------------------------------------------------- */

int main(void)
{
    int Status;

    xil_printf("\r\n");
    xil_printf("========================================\r\n");
    xil_printf(" KV260 AXI DMA V0 LOOPBACK TEST\r\n");
    xil_printf(" DDR -> MM2S -> FIFO -> S2MM -> DDR\r\n");
    xil_printf("========================================\r\n");

    /*
     * Initialize DMA driver/hardware.
     */
    Status = InitDma();

    if (Status != XST_SUCCESS) {

        xil_printf("\r\nV0 TEST FAILED DURING DMA INIT\r\n");

        return XST_FAILURE;
    }

    /*
     * Prepare known source data and clear destination.
     */
    FillBuffers();

    /*
     * Cache maintenance before DMA.
     *
     * TX:
     * Make sure the pattern written by CPU reaches DDR.
     *
     * RX:
     * Make sure there are no dirty cache lines that could later overwrite
     * the data written into DDR by S2MM.
     */
    Xil_DCacheFlushRange((UINTPTR)TxBuffer, TRANSFER_BYTES);
    Xil_DCacheFlushRange((UINTPTR)RxBuffer, TRANSFER_BYTES);

    xil_printf("\r\nStarting DMA transfer...\r\n");

    /*
     * IMPORTANT:
     *
     * Start S2MM first.
     *
     * This arms the destination side before MM2S begins producing
     * AXI4-Stream data.
     *
     * DEVICE_TO_DMA = AXI4-Stream -> memory = S2MM
     */
    Status =
        XAxiDma_SimpleTransfer(&AxiDma,
                               (UINTPTR)RxBuffer,
                               TRANSFER_BYTES,
                               XAXIDMA_DEVICE_TO_DMA);

    if (Status != XST_SUCCESS) {

        xil_printf("ERROR: Failed to start S2MM transfer: %d\r\n",
                   Status);

        DumpDmaStatus(&AxiDma);

        return XST_FAILURE;
    }

    /*
     * Now start MM2S.
     *
     * DMA_TO_DEVICE = memory -> AXI4-Stream = MM2S
     */
    Status =
        XAxiDma_SimpleTransfer(&AxiDma,
                               (UINTPTR)TxBuffer,
                               TRANSFER_BYTES,
                               XAXIDMA_DMA_TO_DEVICE);

    if (Status != XST_SUCCESS) {

        xil_printf("ERROR: Failed to start MM2S transfer: %d\r\n",
                   Status);

        DumpDmaStatus(&AxiDma);

        return XST_FAILURE;
    }

    /*
     * Wait for both channels.
     */
    Status = WaitForDma(&AxiDma);

    if (Status != XST_SUCCESS) {

        xil_printf("\r\nV0 TEST FAILED DURING DMA TRANSFER\r\n");

        return XST_FAILURE;
    }

    xil_printf("DMA transfer completed\r\n");

    DumpDmaStatus(&AxiDma);

    /*
     * S2MM wrote DDR directly.
     *
     * Throw away any old CPU cache lines before reading RxBuffer.
     */
    Xil_DCacheInvalidateRange((UINTPTR)RxBuffer,
                              TRANSFER_BYTES);

    /*
     * Compare what we wrote with what came back.
     */
    Status = CheckData();

    if (Status != XST_SUCCESS) {

        xil_printf("\r\n");
        xil_printf("========================================\r\n");
        xil_printf(" V0 LOOPBACK TEST: FAIL\r\n");
        xil_printf("========================================\r\n");

        return XST_FAILURE;
    }

    xil_printf("\r\n");
    xil_printf("========================================\r\n");
    xil_printf(" V0 LOOPBACK TEST: PASS\r\n");
    xil_printf(" %d bytes copied correctly\r\n",
               TRANSFER_BYTES);
    xil_printf("========================================\r\n");

    return XST_SUCCESS;
}