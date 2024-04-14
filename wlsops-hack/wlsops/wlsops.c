#include "wlsops.h"
#include "WLSINC.h"
#include <linux/string.h>

_adapter *padapter = NULL;
struct ieee80211_local *wls_local = NULL;
struct ieee80211_vif *wls_vif = NULL;
struct ieee80211_vif *t_wls_vif = NULL;
struct ieee80211_hw *wls_hw = NULL;

int wls_hack_init(char * ifname)
{
    struct net_device *dev;
    struct rtw_netdev_priv_indicator *ptr;
    uint8_t counter = 0;
    for_each_netdev(&init_net, dev)
    {
        if (strcmp(dev->name, ifname) == 0)
        {
            ptr = ((struct rtw_netdev_priv_indicator *)netdev_priv(dev));
            printh( "private address: %p, %d\n", ptr->priv, ptr->sizeof_priv );
            padapter = (_adapter *)rtw_netdev_priv(dev);
            if (!padapter) {
                printh("adapter: %p\n", padapter);
                // break;
            }
            counter += 1;
            printh("find wireless realtek device: %s\n", dev->name);
            // break;
        }
        if( dev->ieee80211_ptr ) // is a 802.11 device
        {
            t_wls_vif = wdev_to_ieee80211_vif(dev->ieee80211_ptr);
            if (t_wls_vif) {
                wls_local = (struct ieee80211_local *) wdev_priv(dev->ieee80211_ptr);
                counter += 1;
                wls_hw = &wls_local->hw;
                printh("find 802.11 device: %s\n", dev->name);
                wls_vif = t_wls_vif;
                printh("wls_vif: %p, wls_local: %p, wls_hw:%p\n", t_wls_vif, wls_local, wls_hw);
                // break;
            }
        }
        if (counter > 1){
            break;
        }
    }

    if (!wls_local)
    {
        printh("No 802.11 device found.\n");
        return -1;
    }
    
    return 0;
}

int wls_conf_tx(struct tx_param param)
{

    if (param.if_ind == 0){    // default entry 
        printh("Modify Intel IC\n");                                               
        int ret;
        struct ieee80211_tx_queue_params wls_params = 
        {
            .txop = param.txop,
            .cw_min = param.cw_min,
            .cw_max = param.cw_max,
            .aifs = param.aifs,
            // .acm = false,
            // .uapsd = false,
            // .mu_edca = false
        };
        // printh("wls_vif: %p, wls_local: %p, wls_hw:%p\n", wls_vif, wls_local, wls_hw);
        ret = wls_local->ops->conf_tx(wls_hw, wls_vif, param.ac, &wls_params);
        return ret;
    }
    else{
        printh("Modify Realtek IC\n");
        uint8_t CWMIN, CWMAX, AIFS, aSifsTime;
        uint16_t TXOP;
        uint32_t acParm;

        struct mlme_ext_priv	*pmlmeext = &padapter->mlmeextpriv;
        struct mlme_ext_info	*pmlmeinfo = &(pmlmeext->mlmext_info);

        // if (IsSupported5G(pmlmeext->cur_wireless_mode) || 
        // 	(pmlmeext->cur_wireless_mode & WIRELESS_11_24N) )
        // 	aSifsTime = 16;
        // else
        aSifsTime = 10;
        CWMIN = (uint8_t)param.cw_min ;
        CWMAX = (uint8_t)param.cw_max ;
        AIFS = param.aifs * pmlmeinfo->slotTime + aSifsTime;
        TXOP = param.txop;
        printh("CW-min %d, CW-max %d,AIFS (ms) %d,  TXOP %d\n", CWMIN,CWMAX,AIFS,TXOP);
        acParm = AIFS | (CWMAX << 4) | (CWMIN << 10) | (TXOP << 16);
        printh("AC_param_value %d\n", acParm);
        switch (param.ac)
        {
        case 0:
            padapter->hal_func.set_hw_reg_handler(padapter,HW_VAR_AC_PARAM_VO, (uint8_t*)(&acParm));
            break;
        case 1:
            padapter->hal_func.set_hw_reg_handler(padapter,HW_VAR_AC_PARAM_VI, (uint8_t*)(&acParm));
            break;
        case 2:
            padapter->hal_func.set_hw_reg_handler(padapter,HW_VAR_AC_PARAM_BE, (uint8_t*)(&acParm));
            
            break;
        case 3: 
            padapter->hal_func.set_hw_reg_handler(padapter,HW_VAR_AC_PARAM_BK, (uint8_t*)(&acParm));
            break;
        default:
            padapter->hal_func.set_hw_reg_handler(padapter,HW_VAR_AC_PARAM_BE, (uint8_t*)(&acParm));
            break;
        } 
    }
    return 0;
}
